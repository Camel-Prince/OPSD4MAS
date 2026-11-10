"""Portable entry-point, saved-metric, quota and isolated numerical checks."""
import ast
from collections import defaultdict, Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.run_experiment import load_config, command, check_checkpoint
from scripts.evaluation_records import summarize


def isolated(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class PublicEntrypoints(unittest.TestCase):
    def test_fw4_configuration_and_no_power_variant(self):
        c=load_config(ROOT/'configs/fw4.env')
        cmd=command(c,Path('/tmp/result'))
        self.assertIn('algorithm.opd.facts.future_window=4',cmd)
        self.assertIn('algorithm.opd.power.enabled=false',cmd)
        self.assertIn('actor_rollout_ref.actor.grpo_loss_agg_mode="seq-mean-token-mean"',cmd)
        self.assertIn('trainer.max_actor_ckpt_to_keep=null',cmd)
        self.assertIn('+algorithm.opd.facts.limit_extraction=true',cmd)
        self.assertTrue(any('ppo_micro_batch_size_per_gpu=[2,2]' in x for x in cmd))

    def test_dry_run_six_datasets_without_credentials_or_output(self):
        with tempfile.TemporaryDirectory(prefix='opsd-entry-') as temp:
            dest=Path(temp)/'new'
            env=os.environ.copy()
            env['OPSD_API_KEY']='secret-sentinel-never-log'
            done=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),'evaluate',
                '--checkpoint','/tmp/global_step_100','--dry-run','--output',str(dest)],
                capture_output=True,text=True,env=env,check=True)
            self.assertEqual(done.stdout.count('[DRY RUN]'),6)
            self.assertNotIn('secret-sentinel',done.stdout+done.stderr)
            self.assertNotIn('algorithm.use_opd=true',done.stdout)
            self.assertFalse(dest.exists())

    def test_unicode_path_passed_as_real_unicode(self):
        c=load_config(ROOT/'configs/fw4.env')
        with patch.dict(os.environ,{'MODEL_PATH':'/tmp/模型'}):
            c=load_config(ROOT/'configs/fw4.env')
        cmd=command(c,Path('/tmp/result'))
        self.assertIn('actor_rollout_ref.model.path='+json.dumps(str(Path('/tmp/模型').resolve()),ensure_ascii=False),cmd)

    def test_invalid_config_cannot_execute_shell(self):
        with tempfile.TemporaryDirectory(prefix='opsd-config-') as temp:
            path=Path(temp)/'profile.env'
            path.write_text('MODEL_PATH=$(touch /tmp/opsd-not-allowed)\n')
            with self.assertRaises(ValueError):load_config(path)

    def test_refuse_existing_result_directory(self):
        with tempfile.TemporaryDirectory(prefix='opsd-existing-') as temp:
            sentinel=Path(temp)/'keep.txt';sentinel.write_text('keep')
            run=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),'train','--dry-run','--output',temp],capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertEqual(sentinel.read_text(),'keep')

    def test_checkpoint_requires_both_roles_and_all_ranks(self):
        with tempfile.TemporaryDirectory(prefix='opsd-ckpt-') as temp:
            ck=Path(temp)/'global_step_20'
            for role in ('Solver Agent','Verifier Agent'):
                for rank in range(4):
                    for kind in ('model','optim','extra_state'):
                        p=ck/'actor'/f'old-model_{role}'/f'{kind}_world_size_4_rank_{rank}.pt'
                        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'test')
            self.assertEqual(check_checkpoint(ck,Path('/tmp/new-model'),4),ck.resolve())
            with self.assertRaisesRegex(ValueError,'data.pt'):check_checkpoint(ck,Path('/tmp/new-model'),4,training=True)
            (ck/'actor/old-model_Verifier Agent/model_world_size_4_rank_3.pt').unlink()
            with self.assertRaisesRegex(ValueError,'Missing'):check_checkpoint(ck,Path('/tmp/new-model'),4)

    def test_metrics_count_trajectories_not_agent_rows(self):
        with tempfile.TemporaryDirectory(prefix='opsd-metric-') as temp:
            p=Path(temp)/'predictions.jsonl'
            rows=[]
            for qi,correct in enumerate(([True,False,False,False],[True,True,True,True])):
                for ti,value in enumerate(correct):
                    for _ in range(5):
                        rows.append({'data_source':'fixture','uid':str(qi),'traj_uid':f'{qi}-{ti}','pass':value})
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=summarize(p,2)
            self.assertEqual(result['complete_trajectories'],8)
            self.assertEqual(result['pass@1'],.625)
            self.assertEqual(result['pass@4'],1)

    def test_incomplete_trajectory_group_rejected(self):
        with tempfile.TemporaryDirectory(prefix='opsd-metric-') as temp:
            p=Path(temp)/'predictions.jsonl'
            p.write_text(json.dumps({'data_source':'x','uid':'q','traj_uid':'t','pass':True})+'\n')
            with self.assertRaisesRegex(ValueError,'exactly 4'):summarize(p)

    def test_filler_hooks_are_inert_even_if_parent_environment_has_signal(self):
        from scripts.gpu_filler_control import pause,resume
        with patch.dict(os.environ,{'FILLER_SIGNAL_FILE':'/tmp/do-not-use-filler'}):
            self.assertFalse(pause());self.assertFalse(resume())


class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=isolated('_opsd_release_fact_pipeline',ROOT/'verl/trainer/ppo/opd_fact_cards.py')

    def test_api_retries_use_shared_limiter_and_no_sdk_retries(self):
        calls=[]
        class Client:
            def __init__(self):self.chat=types.SimpleNamespace(completions=self)
            def with_options(self,**kw):calls.append(('sdk',kw));return self
            def create(self,**kw):
                calls.append(('call',kw))
                if sum(x[0]=='call' for x in calls)==1:raise RuntimeError('temporary transport error')
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{}'),finish_reason='stop')],usage={'total_tokens':7})
        cfg={'model_name':'fake','endpoint':'','max_retries':1,'limit_extraction':True,'max_tokens':10,'relation_tpm_limit':10000,'relation_rpm_limit':1000}
        pipeline=self.m.FactCardPipeline(cfg,client=Client())
        class Limiter:
            def __init__(self):self.calls=[]
            def acquire(self,tokens):self.calls.append(tokens);return len(self.calls)
            def reconcile(self,*args):pass
        limiter=Limiter();pipeline.relation_rate_limiter=limiter
        result=pipeline._call_json('extraction','hash','prompt',[])
        self.assertIsNone(result.error)
        self.assertEqual(len(limiter.calls),2)
        self.assertEqual([kw for label,kw in calls if label=='sdk'],[{'max_retries':0}]*2)

    def test_facts_use_future_window_four(self):
        m=self.m
        records=[m.ResponseRecord(response_uid=f'r{i}',query_uid='q',trajectory_uid='t',agent_id='Solver Agent',sequence_index=i,agent_turn_index=i,raw_step_index=i,response_text='x',comment_last_agent='',student_prompt_messages=[],query_text='q') for i in range(5)]
        targets=m.build_targets_for_source(records[-1],records,4)
        self.assertEqual([t.future_distance for t in targets],[1,2,3,4])

    def test_fake_api_prompts_exclude_reward_and_ground_truth_fields(self):
        p=self.m.FactCardPipeline({'model_name':'fake','endpoint':''},client=object())
        prompt=p._extraction_prompt([{'record_id':'r','problem':'P','evaluated_response':'R','comment':'C'}])
        self.assertNotIn('"ground_truth"',prompt)
        self.assertNotIn('"reward"',prompt)


class IsolatedNumericalChecks(unittest.TestCase):
    """Execute actual source function bodies with real CPU torch; no Ray/runtime stubs.

    This isolates math from unavailable distributed imports. Full import-based tests
    remain in tests/trainer/ppo and run in the documented target/CI environment.
    """
    @classmethod
    def setUpClass(cls):
        import torch,numpy as np,math
        from typing import Dict,List,Optional,Union,Sequence,Tuple,Any
        from contextlib import contextmanager
        env=locals().copy()
        util=types.ModuleType('_opsd_math_primitives')
        util.__dict__.update(env)
        util.F=torch.nn.functional
        util.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE=False
        def load_functions(path,namespace):
            tree=ast.parse(path.read_text())
            future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
            tree=ast.Module(body=[future]+[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))],type_ignores=[])
            exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
        load_functions(ROOT/'verl/utils/torch_functional.py',util.__dict__)
        core=dict(env,verl_F=util,defaultdict=defaultdict,Counter=Counter)
        load_functions(ROOT/'verl/trainer/ppo/core_algos.py',core)
        cls.core=core;cls.torch=torch

    def test_grpo_response_mean_and_opd_token_mean_differ(self):
        t=self.torch;c=self.core
        mask=t.tensor([[1.,0.,0.],[1.,1.,1.]])
        adv=t.tensor([[1.,0.,0.],[-1.,-1.,-1.]])
        z=t.zeros_like(adv)
        loss,*_=c['compute_policy_loss'](old_log_prob=z,log_prob=z,advantages=adv,response_mask=mask,cliprange=.2,loss_agg_mode='seq-mean-token-mean')
        t.testing.assert_close(loss,t.tensor(0.))
        t.testing.assert_close(c['agg_loss'](-adv,mask,'token-mean'),t.tensor(.5))

    def test_opd_gradient_only_through_current_probabilities(self):
        t=self.torch;c=self.core
        current=t.tensor([[-1.,-2.]],requires_grad=True)
        coefficient=t.tensor([[.2,-.1]])
        loss=c['agg_loss'](-coefficient*current,t.ones_like(current),'token-mean')
        loss.backward()
        t.testing.assert_close(current.grad,t.tensor([[-.1,.05]]))


if __name__=='__main__':unittest.main()
