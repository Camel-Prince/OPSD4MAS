"""Portable foreground FW4 launchers. Dry runs perform no model/API work or output writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COUNTS = {'AMC23':83, 'AIME24':30, 'AIME25':30, 'MATH-500':500, 'Olympiad-Bench':675, 'Minerva':272}


def load_config(path):
    values = {}
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            raise ValueError(f'{path}:{number}: expected KEY=value')
        key, value = line.split('=', 1)
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', key):
            raise ValueError(f'{path}:{number}: invalid key')
        if key.endswith('_KEY'):
            raise ValueError(f'{path}:{number}: credentials belong in environment variables, not the profile')
        parts = shlex.split(value, comments=True)
        if len(parts)>1:
            raise ValueError(f'{path}:{number}: quote values containing spaces')
        values[key] = os.environ.get(key, parts[0] if parts else '')
    return values


def absolute(value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def integer(c, key, minimum=1):
    value=int(c[key])
    if value<minimum:
        raise ValueError(f'{key} must be >= {minimum}')
    return value


def check_model(c):
    model=absolute(c['MODEL_PATH'])
    cfg=json.loads((model/'config.json').read_text())
    if cfg.get('model_type') != 'qwen3':
        raise ValueError('FW4 reference profile requires Qwen3 model config')
    if not (model/'tokenizer_config.json').is_file():
        raise ValueError('Missing tokenizer_config.json')
    if (model/'model.safetensors.index.json').is_file():
        shards=set(json.loads((model/'model.safetensors.index.json').read_text())['weight_map'].values())
    else:
        shards={'model.safetensors'}
    if not shards or any(not (model/n).is_file() or (model/n).stat().st_size==0 for n in shards):
        raise ValueError('Incomplete base-model weights')
    return model


def check_checkpoint(path, model, world, training=False):
    path=absolute(path)
    if not re.fullmatch(r'global_step_\d+',path.name):
        raise ValueError('Use a global_step_N checkpoint directory')
    for role in ('Solver Agent','Verifier Agent'):
        exact=path/'actor'/f'{model.name}_{role}'
        matches=[exact] if exact.is_dir() else sorted((path/'actor').glob(f'*_{role}'))
        if len(matches)!=1:
            raise ValueError(f'Expected exactly one {role} checkpoint under {path}/actor')
        for rank in range(world):
            for kind in ('model','optim','extra_state'):
                f=matches[0]/f'{kind}_world_size_{world}_rank_{rank}.pt'
                if not f.is_file() or f.stat().st_size==0:
                    raise ValueError(f'Missing or empty checkpoint shard: {f}')
    if training and not (path/'data.pt').is_file():
        raise ValueError('Exact training resume requires checkpoint data.pt')
    return path


def check_inputs(c, datasets=None):
    import pyarrow.parquet as pq
    data=absolute(c['DATA_DIR'])
    manifest=json.loads((data/'manifest.json').read_text())
    names=['train','validation'] if datasets is None else ['train']+list(datasets)
    for name in names:
        path=data/(f'{name}.parquet' if name in ('train','validation') else f'{name}/test.parquet')
        pf=pq.ParquetFile(path)
        if not {'prompt','env_kwargs','data_source','reward_model','extra_info'} <= set(pf.schema_arrow.names):
            raise ValueError(f'Invalid prepared data schema: {path}')
        source=manifest['training'] if name=='train' else manifest['validation'] if name=='validation' else manifest['benchmarks'][name]
        wanted=source.get('prepared_sha256') or source.get('sha256')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=wanted:
            raise ValueError(f'Prepared file differs from its manifest: {path}')
        expected=COUNTS.get(name)
        if expected is not None and pf.metadata.num_rows != expected:
            raise ValueError(f'{name}: expected {expected} rows, found {pf.metadata.num_rows}')
        if pf.metadata.num_rows==0:
            raise ValueError(f'Empty data: {path}')
    if pq.ParquetFile(data/'train.parquet').metadata.num_rows<integer(c,'TRAIN_BATCH_SIZE'):
        raise ValueError('Train loader needs at least one full train batch, including validation-only runs')


def command(c, output, dataset=None, checkpoint=None):
    evaluate=dataset is not None
    model=str(absolute(c['MODEL_PATH']))
    data=absolute(c['DATA_DIR'])
    prompt=14336 if evaluate else integer(c,'MAX_PROMPT_LENGTH')
    response=6144 if evaluate else integer(c,'MAX_RESPONSE_LENGTH')
    d={
      'algorithm.adv_estimator':'grpo' if evaluate else 'opd_grpo_joint_loss',
      'algorithm.use_opd':not evaluate,'algorithm.group_by_agent_id':True,
      'algorithm.opd.version':'v4','algorithm.opd.v4_comment_enabled':True,
      'algorithm.opd.use_judger':False,'algorithm.opd.use_facts':not evaluate,
      'algorithm.opd.power.enabled':False,
      'algorithm.opd.max_teacher_context_length':integer(c,'MAX_TEACHER_CONTEXT_LENGTH'),
      'algorithm.opd.loss_weight_init':float(c['OPD_LOSS_WEIGHT_INIT']),
      'algorithm.opd.loss_weight_anneal_rate':float(c['OPD_LOSS_WEIGHT_ANNEAL_RATE']),
      'algorithm.opd.loss_weight_min':float(c['OPD_LOSS_WEIGHT_MIN']),
      'algorithm.opd.loss_grpo_outcome_weight':1.0,
      'data.train_files':str(data/'train.parquet'),
      'data.val_files':str(data/f'{dataset}/test.parquet') if evaluate else str(data/'validation.parquet'),
      'data.train_batch_size':integer(c,'TRAIN_BATCH_SIZE'), 'data.val_batch_size':integer(c,'VAL_BATCH_SIZE'),
      'data.max_prompt_length':prompt,'data.max_response_length':response,
      'data.filter_overlong_prompts':False,'data.truncation':'middle','data.return_raw_chat':True,
      '+data.apply_chat_template_kwargs.enable_thinking':False,'+data.dataloader_num_workers':0,
      'actor_rollout_ref.model.path':model,'actor_rollout_ref.actor.optim.lr':1e-6,
      'actor_rollout_ref.model.use_remove_padding':True,
      'actor_rollout_ref.model.enable_gradient_checkpointing':True,
      'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':None,
      '+agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu':[integer(c,'ACTOR_MICRO_BATCH_SIZE')]*2,
      'actor_rollout_ref.actor.ppo_mini_batch_size':256,
      'actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size':True,
      'actor_rollout_ref.actor.ppo_mini_update_num':1,'actor_rollout_ref.actor.ppo_epochs':integer(c,'PPO_EPOCHS'),
      'actor_rollout_ref.actor.use_dynamic_bsz':False,
      'actor_rollout_ref.actor.use_kl_loss':False,'actor_rollout_ref.actor.entropy_coeff':0.0,
      'actor_rollout_ref.actor.grpo_loss_agg_mode':'seq-mean-token-mean',
      'actor_rollout_ref.actor.opd_loss_agg_mode':'token-mean',
      'actor_rollout_ref.actor.use_invalid_action_penalty':True,
      'actor_rollout_ref.actor.invalid_action_penalty_coef':0.1,
      'actor_rollout_ref.actor.fsdp_config.param_offload':True,
      'actor_rollout_ref.actor.fsdp_config.optimizer_offload':True,
      'actor_rollout_ref.rollout.name':'sglang','actor_rollout_ref.rollout.n':1,
      'actor_rollout_ref.rollout.tensor_model_parallel_size':1,
      'actor_rollout_ref.rollout.gpu_memory_utilization':0.5,
      'actor_rollout_ref.rollout.enforce_eager':True,'actor_rollout_ref.rollout.free_cache_engine':True,
      'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':1 if evaluate else integer(c,'LOG_PROB_MICRO_BATCH_SIZE'),
      'actor_rollout_ref.rollout.val_kwargs.do_sample':True,'actor_rollout_ref.rollout.val_kwargs.n':1,
      'actor_rollout_ref.rollout.val_kwargs.temperature':0.6,'actor_rollout_ref.rollout.val_kwargs.top_p':0.95,
      'env.env_name':'math','env.seed':integer(c,'SEED',0),'env.rollout.n':integer(c,'GROUP_SIZE'),
      'env.rollout.val_n':4,'agent.agent_ids':['Solver Agent','Verifier Agent'],
      'agent.model_ids':[model,model],'agent.external_agent_ids':[],
      'agent.model_sharing':False,'agent.orchestra_type':'math',
      'agent.orchestra.math.max_loop_num':integer(c,'MAX_LOOP_NUM'),
      'trainer.n_gpus_per_node':integer(c,'NUM_GPUS'),'trainer.nnodes':1,
      'trainer.logger':['console'],'trainer.project_name':'OPSD4MAS',
      'trainer.experiment_name':output.name,'trainer.default_local_dir':str(output),
      'trainer.resume_mode':'resume_path' if checkpoint else 'disable',
      'trainer.resume_from_path':str(absolute(checkpoint)) if checkpoint else None,
      'trainer.del_local_ckpt_after_load':False,
      'trainer.max_actor_ckpt_to_keep':None,'trainer.max_critic_ckpt_to_keep':None,
      'trainer.val_only':evaluate,'trainer.val_before_train':True,
      'trainer.total_epochs':integer(c,'TOTAL_EPOCHS'),
      'trainer.total_training_steps':integer(c,'TOTAL_TRAINING_STEPS'),
      'trainer.save_freq':integer(c,'SAVE_FREQ'),'trainer.test_freq':integer(c,'TEST_FREQ'),
      'trainer.validation_data_dir':str(output/'predictions'),
      'ray_init.num_cpus':min(32,os.cpu_count() or 8),'hydra.run.dir':str(output/'hydra'),
    }
    if not evaluate:
        facts={
          'selection_mode':'fact_card_loo','future_window':integer(c,'FACT_FUTURE_WINDOW'),
          'max_facts_per_comment':integer(c,'FACT_MAX_FACTS_PER_COMMENT'),
          'max_candidates_per_target':integer(c,'FACT_MAX_CANDIDATES'),
          'max_selected_joint_loss':integer(c,'FACT_MAX_SELECTED'),
          'loo_variant_batch_size':integer(c,'FACT_LOO_VARIANT_BATCH_SIZE'),
          'min_effect':float(c['FACT_MIN_EFFECT']),'min_precision':float(c['FACT_MIN_PRECISION']),
          'min_net_alignment':float(c['FACT_MIN_NET_ALIGNMENT']),'conflict_weight':float(c['FACT_CONFLICT_WEIGHT']),
          'endpoint':c['OPSD_API_BASE_URL'],'model_name':c['OPSD_API_MODEL'],'api_key_env':c['OPSD_API_KEY_ENV'],
          'relation_tpm_limit':int(c['OPSD_API_TPM'] or 100000),
          'relation_rpm_limit':int(c['OPSD_API_RPM'] or 30),
          'relation_max_concurrent':int(c['OPSD_API_CONCURRENCY'] or 2),
          'max_concurrent':int(c['OPSD_API_CONCURRENCY'] or 2),
          'relation_facts_per_request':12,'relation_trajectories_per_request':4,
          'relation_max_input_tokens':24000,'inter_stage_wait_seconds':60,
          'max_tokens':integer(c,'OPSD_API_MAX_TOKENS'),'max_retries':2,
          'artifact_dir':str(output/'fact_artifacts'),'cache_dir':None,
          'final_save_freq':integer(c,'SAVE_FREQ'),'save_trajectories':True,
          'trajectory_save_freq':integer(c,'SAVE_FREQ'),'save_api_audit':False,
          'privileged_token_budget':512,
        }
        d.update({'algorithm.opd.facts.'+k:v for k,v in facts.items()})
        d['+algorithm.opd.facts.limit_extraction']=True
    return [sys.executable,'-u','-m','verl.trainer.main_ppo']+[f'{k}={json.dumps(v,ensure_ascii=False,separators=(",",":"))}' for k,v in d.items()]


def run_logged(cmd, out, env, config):
    out.mkdir(parents=True, exist_ok=False)
    (out/'launch-config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'source-provenance.json').write_bytes((ROOT/'docs/source-provenance.json').read_bytes())
    (out/'data-manifest.json').write_bytes((absolute(config['DATA_DIR'])/'manifest.json').read_bytes())
    (out/'launch-command.json').write_text(json.dumps(cmd,ensure_ascii=False,indent=2)+'\n')
    with (out/'pip-freeze.txt').open('w') as versions:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=versions,check=True)
    print(f'[RUN] {out}\n[LOG] {out / "console.log"}',flush=True)
    with (out/'console.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               text=True,bufsize=1,start_new_session=True)
        try:
            for line in child.stdout:
                sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
            code=child.wait()
        except KeyboardInterrupt:
            os.killpg(child.pid,signal.SIGINT)
            child.wait()
            raise
    (out/'exit-code.txt').write_text(str(code)+'\n')
    if code:
        raise RuntimeError(f'Run exited {code}; retained log: {out / "console.log"}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('profile',nargs='?',default='configs/fw4.env')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--datasets',nargs='+',default=['all'])
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    c=load_config(absolute(a.profile))
    if a.mode=='evaluate' and not a.checkpoint:
        p.error('evaluate requires --checkpoint /path/to/global_step_N')
    if a.mode=='train' and a.checkpoint:
        p.error('use --resume-from for training')
    if a.mode=='evaluate' and a.resume_from:
        p.error('use --checkpoint for evaluation')
    names=list(COUNTS) if a.datasets==['all'] else a.datasets
    if any(n not in COUNTS for n in names) or len(set(names))!=len(names):
        p.error('datasets must be unique names from '+', '.join(COUNTS))
    checkpoint=a.checkpoint or a.resume_from
    run_id='fw4-'+a.mode+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    output=absolute(a.output) if a.output else absolute(c['OUTPUT_ROOT'])/run_id
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing output: {output}')
    jobs=[(name,output/name) for name in names] if a.mode=='evaluate' else [(None,output)]
    if a.dry_run:
        for dataset, out in jobs:
            print('[DRY RUN]',shlex.join(command(c,out,dataset,checkpoint)))
        return
    world=integer(c,'NUM_GPUS')
    if world!=4:
        raise ValueError('This release profile targets four GPUs; validate other topologies separately')
    model=check_model(c)
    check_inputs(c,names if a.mode=='evaluate' else None)
    if checkpoint:
        check_checkpoint(checkpoint,model,world,training=a.mode=='train')
    if a.mode=='train':
        for k in ('OPSD_API_BASE_URL','OPSD_API_MODEL','OPSD_API_TPM','OPSD_API_RPM','OPSD_API_CONCURRENCY'):
            if not c[k]: raise ValueError(f'Set {k} before training')
        if not os.environ.get(c['OPSD_API_KEY_ENV']):
            raise ValueError('Set the API key environment variable; it will not be printed')
        for k in ('OPSD_API_TPM','OPSD_API_RPM','OPSD_API_CONCURRENCY'):integer(c,k)
    from scripts.check_environment import check_gpu
    check_gpu(world)
    env=os.environ.copy()
    for key in ('BASH_ENV','PYTHONHOME','FILLER_SIGNAL_FILE','PW_DEBUG','PW_DEBUG_STEPS'):
        env.pop(key,None)
    env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',RAY_ADDRESS='local',HYDRA_FULL_ERROR='1')
    env['PYTHONPATH']=str(ROOT)
    cache=Path(env.get('TMPDIR','/tmp'))/('opsd4mas-'+uuid.uuid4().hex[:12])
    for key,suffix in [('FLASHINFER_WORKSPACE_BASE','flashinfer'),('TRITON_CACHE_DIR','triton'),
                       ('TORCHINDUCTOR_CACHE_DIR','inductor'),('TORCH_EXTENSIONS_DIR','extensions')]:
        env[key]=str(cache/suffix)
    if a.mode=='evaluate':output.mkdir(parents=True,exist_ok=False)
    for dataset,out in jobs:
        run_logged(command(c,out,dataset,checkpoint),out,env,c)
        if dataset:
            from scripts.evaluation_records import summarize
            result=summarize(out/'predictions/predictions.jsonl',COUNTS[dataset])
            result['dataset']=dataset
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'[RESULT] {dataset}: pass@1={result["pass@1"]:.4%}, pass@4={result["pass@4"]:.4%}',flush=True)
    if a.mode=='evaluate':
        results=[json.loads((output/n/'result.json').read_text()) for n in names]
        (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        lines=['| Dataset | Questions | pass@1 | pass@4 |','|---|---:|---:|---:|']
        lines += [f'| {r["dataset"]} | {r["questions"]} | {r["pass@1"]:.4%} | {r["pass@4"]:.4%} |' for r in results]
        (output/'results.md').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines))


if __name__=='__main__':
    try: main()
    except (ValueError,FileNotFoundError,RuntimeError) as error:
        print('[ERROR]',error,file=sys.stderr)
        sys.exit(1)
