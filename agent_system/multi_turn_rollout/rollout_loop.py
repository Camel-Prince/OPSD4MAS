# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import os
import json
import datetime as dt
import hashlib
import sys
import time
from copy import deepcopy
from pathlib import Path
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy, filter_group_data, preprocess_batch
from agent_system.environments import EnvironmentManagerBase
from agent_system.agent import BaseOrchestra
from typing import List, Dict, Any, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def _stable_response_uid(global_step, trajectory_uid, raw_step_index, agent_id):
    payload = f"{global_step}\x1f{trajectory_uid}\x1f{raw_step_index}\x1f{agent_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

def _dump_pw_debug_json(
    total_batch_list,
    effective_batch,
    ground_truths,
    questions,
    fact_pools,
    episode_rewards,
    traj_uid,
    success,
    tokenizers,
    save_dir,
    judger_records=None,
):
    """When PW_DEBUG=True, dump structured trajectory data as JSON for visualization."""
    import sys

    os.makedirs(save_dir, exist_ok=True)

    # Pick a tokenizer for decoding teacher tensors
    _tk = list(tokenizers.values())[0] if isinstance(tokenizers, dict) else tokenizers

    trajectories = []
    for env_idx, steps in enumerate(total_batch_list):
        if not steps:
            continue
        entry = {
            "traj_uid": str(traj_uid[env_idx]),
            "question": questions[env_idx] if questions else "",
            "ground_truth": str(ground_truths[env_idx]) if ground_truths else "",
            "episode_reward": float(episode_rewards[env_idx]) if env_idx < len(episode_rewards) else 0.0,
            "pass": float(success.get("success_rate", [0])[env_idx]) if success and env_idx < len(success.get("success_rate", [])) else 0.0,
            "steps": [],
            "facts": [],
        }

        for step_idx, data in enumerate(steps):
            step = {
                "step_idx": step_idx,
                "agent_id": str(data.get("agent_id", "")),
                "text_response": str(data.get("text_response", "")),
                "agent_active_mask": bool(data.get("agent_active_mask", False)),
                "active_masks": bool(data.get("active_masks", False)),
            }
            entry["steps"].append(step)

        # Attach facts if available
        if fact_pools and env_idx in fact_pools:
            for f in fact_pools[env_idx]:
                entry["facts"].append({"step_idx": f.step_idx, "text": f.text})

        trajectories.append(entry)

    # Decode teacher context for the first few OPD samples (same as OPD_INSPECT)
    teacher_samples = []
    try:
        pad = _tk.pad_token_id
        n_inspect = min(3, len(effective_batch))
        for i in range(len(effective_batch)):
            if len(teacher_samples) >= n_inspect:
                break
            eb = effective_batch[i]
            if "teacher_input_ids" not in eb:
                continue
            sample = {
                "agent_id": str(eb.get("agent_id", "")),
                "traj_uid": str(eb.get("traj_uid", "")),
                "student_prompt": "",
                "privileged_context": "",
            }
            # Decode student prompt — use "prompts" tensor directly
            try:
                prompt_tensor = eb["prompts"]
                if pad is not None:
                    prompt_tensor = prompt_tensor[prompt_tensor != pad]
                sample["student_prompt"] = _tk.decode(prompt_tensor.tolist(), skip_special_tokens=False)
            except Exception:
                pass
            # Decode teacher suffix (teacher_input_ids minus student prompt)
            try:
                teacher_ids = eb["teacher_input_ids"][eb["teacher_attention_mask"].bool()]
                # Student prompt length = # of non-pad tokens in "prompts"
                sp = eb["prompts"]
                student_prompt_len = int((sp != pad).sum()) if pad is not None else len(sp)
                suffix_ids = teacher_ids[student_prompt_len:]
                sample["privileged_context"] = _tk.decode(suffix_ids.tolist(), skip_special_tokens=False)
                sample["teacher_prompt"] = _tk.decode(teacher_ids.tolist(), skip_special_tokens=False)
            except Exception:
                pass
            teacher_samples.append(sample)
    except Exception as e:
        print(f"[PW_DEBUG] teacher decode failed: {e}", file=sys.stderr)

    output = {
        "timestamp": dt.datetime.now().isoformat(),
        "num_trajectories": len(trajectories),
        "trajectories": trajectories,
        "teacher_samples": teacher_samples,
        "judger_records": judger_records or [],
    }

    basename = os.environ.get("PW_DEBUG_LOG_BASENAME", dt.datetime.now().strftime("step_%Y%m%d_%H%M%S"))
    path = os.path.join(save_dir, f"{basename}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[PW_DEBUG] saved {len(trajectories)} trajectories → {path}", file=sys.stderr, flush=True)


def _dump_training_trajectories_jsonl_zst(
    total_batch_list,
    ground_truths,
    questions,
    fact_pools,
    episode_rewards,
    traj_uid,
    success,
    global_step,
    output_dir,
    filename_prefix,
):
    """Atomically save every grouped training trajectory as zstd-compressed JSONL."""
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError(
            "SAVE_ALL_TRAJECTORIES=True requires the 'zstandard' package"
        ) from exc

    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{filename_prefix}_step{global_step}.jsonl.zst"
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    records = []
    for env_idx, steps in enumerate(total_batch_list):
        facts = []
        if fact_pools and env_idx in fact_pools:
            facts = [
                {"step_idx": int(fact.step_idx), "text": str(fact.text)}
                for fact in fact_pools[env_idx]
            ]
        records.append({
            "global_step": int(global_step),
            "trajectory_uid": str(traj_uid[env_idx]),
            "question": str(questions[env_idx]) if questions is not None else "",
            "ground_truth": str(ground_truths[env_idx]) if ground_truths is not None else "",
            "episode_reward": float(episode_rewards[env_idx]),
            "pass": bool(success["success_rate"][env_idx]),
            "steps": [{
                "step_idx": step_idx,
                "agent_id": str(data.get("agent_id", "")),
                "text_response": str(data.get("text_response", "")),
                "agent_active_mask": bool(data.get("agent_active_mask", False)),
                "active_masks": bool(data.get("active_masks", False)),
            } for step_idx, data in enumerate(steps)],
            "facts": facts,
        })

    lines = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    temp_path.write_bytes(zstd.ZstdCompressor().compress(lines))
    os.replace(temp_path, output_path)
    print(
        f"[TrajectoryDump] saved {len(records)} trajectories → {output_path}",
        file=sys.stderr,
        flush=True,
    )


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ground_truths=None,
            questions=None,
            is_train: bool = True,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.

        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment

        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        # Resume GPU filler — fact extraction + teacher construction are CPU/API-bound, GPU idle
        try:
            from scripts.gpu_filler_control import resume as gpu_filler_resume
            gpu_filler_resume()
        except Exception:
            pass

        # success_rate = {}
        # for key, value in success.items():
        #     success_rate[key] = np.mean(value)

        effective_batch = []
        for bs in range(batch_size):
            num_steps = len(total_batch_list[bs])
            # sum the rewards for each data in total_batch_list[bs]
            for step_i, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    # for key, value in success_rate.items():
                    #     data[key] = value
                    # pass
                    data['pass'] = success['success_rate'][bs]
                    # Mark whether this is the last step in the trajectory
                    data['is_last_step'] = (step_i == num_steps - 1)

                    effective_batch.append(data)

        fact_card_mode = is_train and getattr(self, "_fact_card_pipeline_enabled", False)
        if fact_card_mode:
            from verl.trainer.ppo.opd_fact_cards import ResponseRecord, extract_comment

            response_records = []
            for steps in total_batch_list:
                for data in steps:
                    if not data.get("active_masks", False) or data.get("agent_id") == "Judger Agent":
                        continue
                    response_records.append(ResponseRecord(
                        response_uid=str(data["response_uid"]),
                        query_uid=str(data["uid"]),
                        trajectory_uid=str(data["traj_uid"]),
                        agent_id=str(data["agent_id"]),
                        sequence_index=int(data["sequence_index"]),
                        agent_turn_index=int(data["agent_turn_index"]),
                        raw_step_index=int(data["raw_step_index"]),
                        response_text=str(data.get("text_response", "")),
                        comment_last_agent=extract_comment(str(data.get("text_response", ""))),
                        student_prompt_messages=list(data.get("student_prompt_messages", [])),
                        query_text=str(data.get("query_text", "")),
                    ))
            rewards_by_trajectory = {
                str(traj_uid[index]): float(episode_rewards[index]) for index in range(batch_size)
            }
            correctness_by_trajectory = {
                str(traj_uid[index]): bool(success["success_rate"][index]) for index in range(batch_size)
            }
            records_payload = json.dumps(
                [record.__dict__ for record in response_records], ensure_ascii=False
            )
            outcomes_payload = json.dumps({
                "rewards": rewards_by_trajectory,
                "correctness": correctness_by_trajectory,
            }, ensure_ascii=False)
            trajectory_artifacts = []
            for env_index, steps in enumerate(total_batch_list):
                trajectory_artifacts.append({
                    "query_uid": str(steps[0].get("uid", "")) if steps else "",
                    "trajectory_uid": str(traj_uid[env_index]),
                    "question": str(questions[env_index]) if questions is not None else "",
                    "ground_truth": str(ground_truths[env_index]) if ground_truths is not None else "",
                    "episode_reward": float(episode_rewards[env_index]),
                    "pass": bool(success["success_rate"][env_index]),
                    "responses": [{
                        "response_uid": str(data.get("response_uid", "")),
                        "agent_id": str(data.get("agent_id", "")),
                        "sequence_index": int(data.get("sequence_index", step_index)),
                        "agent_turn_index": int(data.get("agent_turn_index", 0)),
                        "raw_step_index": int(data.get("raw_step_index", step_index)),
                        "active": bool(data.get("active_masks", False)),
                        "response_text": str(data.get("text_response", "")),
                        "comment_last_agent": extract_comment(str(data.get("text_response", ""))),
                        "student_prompt_messages": list(data.get("student_prompt_messages", [])),
                    } for step_index, data in enumerate(steps)],
                })
            trajectories_payload = json.dumps(trajectory_artifacts, ensure_ascii=False)
            for index, data in enumerate(effective_batch):
                data["fact_records_payload"] = records_payload if index == 0 else ""
                data["fact_outcomes_payload"] = outcomes_payload if index == 0 else ""
                data["fact_trajectories_payload"] = trajectories_payload if index == 0 else ""

        # OPD: construct teacher prompt data with privileged future context
        fact_pools = None
        use_opd = is_train and str(getattr(self.config.algorithm, 'use_opd', False)).strip().lower() in ("true", "1")
        if use_opd and not fact_card_mode and hasattr(self, 'multiagent_orchestra'):
            from verl.trainer.ppo.opd_teacher import construct_teacher_data, pad_teacher_data
            opd_config = getattr(self.config.algorithm, 'opd', {})
            max_teacher_ctx = getattr(opd_config, 'max_teacher_context_length', 4096)
            version = getattr(opd_config, 'version', 'v1')
            agents_to_wg_mapping = self.multiagent_orchestra.agents_to_wg_mapping

            # New OPD options (改进1, 改进3)
            # Normalize str "False"/"True" (from shell env vars) to real bools.
            use_judger = str(getattr(opd_config, 'use_judger', False)).strip().lower() in ("true", "1")
            use_facts = str(getattr(opd_config, 'use_facts', False)).strip().lower() in ("true", "1")

            # One-shot fact extraction per trajectory (改进3, new design)
            fact_pools = None
            if use_facts and self._fact_extractor is not None:
                import sys
                from verl.trainer.ppo.opd_teacher import _extract_judgment_from_trajectory
                n_envs = len(total_batch_list)
                print(f"[Facts] extracting from {n_envs} trajectories...",
                      file=sys.stderr, flush=True)
                fact_pools = {}
                for env_idx, steps in enumerate(total_batch_list):
                    if not steps:
                        continue
                    try:
                        judgment_text = _extract_judgment_from_trajectory(
                            total_batch_list, env_idx
                        ) if use_judger else ""
                        fact_pools[env_idx] = self._fact_extractor.extract_all(
                            trajectory_steps=steps,
                            judger_j=judgment_text,
                        )
                        if (env_idx + 1) % max(1, n_envs // 10) == 0 or env_idx == n_envs - 1:
                            print(f"[Facts] {env_idx + 1}/{n_envs} trajectories done",
                                  file=sys.stderr, flush=True)
                    except Exception:
                        fact_pools[env_idx] = []

                total_facts = sum(len(p) for p in fact_pools.values())
                print(f"[Facts] extraction done: {total_facts} facts from {len(fact_pools)} trajectories",
                      file=sys.stderr, flush=True)

            teacher_items = construct_teacher_data(
                total_batch_list=total_batch_list,
                effective_batch=effective_batch,
                tokenizers=self.tokenizer,
                agents_to_wg_mapping=agents_to_wg_mapping,
                max_teacher_context_length=max_teacher_ctx,
                version=version,
                use_judger=use_judger,
                use_facts=use_facts,
                fact_pools=fact_pools,
                fact_filter=self._fact_filter,
            )
            first_tokenizer = next(iter(self.tokenizer.values()))
            pad_token_id = first_tokenizer.pad_token_id if first_tokenizer.pad_token_id is not None else 0
            teacher_batch_data = pad_teacher_data(teacher_items, pad_token_id=pad_token_id)

            # Store teacher tensors in effective_batch items for collation
            for i, data_item in enumerate(effective_batch):
                data_item['teacher_input_ids'] = teacher_batch_data['teacher_input_ids'][i]
                data_item['teacher_attention_mask'] = teacher_batch_data['teacher_attention_mask'][i]
                data_item['teacher_position_ids'] = teacher_batch_data['teacher_position_ids'][i]

        # --- PW_DEBUG: dump structured JSON for visualization ---
        # FactCard LOO consumes trajectories in memory; persistent JSON dumps
        # are debug-only and must be explicitly opted into.
        if os.environ.get("PW_DEBUG", "").strip().lower() in {"true", "1"}:
            judger_records = None
            if hasattr(self, 'multiagent_orchestra'):
                judger_agent = self.multiagent_orchestra.agents.get("Judger Agent")
                if judger_agent and hasattr(judger_agent, 'debug_records'):
                    judger_records = list(judger_agent.debug_records)
            _dump_pw_debug_json(
                total_batch_list=total_batch_list,
                effective_batch=effective_batch,
                ground_truths=ground_truths,
                questions=questions,
                fact_pools=fact_pools,
                episode_rewards=episode_rewards,
                traj_uid=traj_uid,
                success=success,
                tokenizers=self.tokenizer,
                save_dir=os.environ.get("PW_DEBUG_DIR", "./pw_debug_dumps"),
                judger_records=judger_records,
            )

        if is_train and os.environ.get("SAVE_ALL_TRAJECTORIES", "False").strip().lower() in {
            "true", "1"
        }:
            trajectory_dump_dir = os.environ.get("TRAJECTORY_DUMP_DIR")
            if not trajectory_dump_dir:
                raise ValueError(
                    "TRAJECTORY_DUMP_DIR must be set when SAVE_ALL_TRAJECTORIES=True"
                )
            _dump_training_trajectories_jsonl_zst(
                total_batch_list=total_batch_list,
                ground_truths=ground_truths,
                questions=questions,
                fact_pools=fact_pools,
                episode_rewards=episode_rewards,
                traj_uid=traj_uid,
                success=success,
                global_step=getattr(self, "_current_global_step", 0),
                output_dir=trajectory_dump_dir,
                filename_prefix=os.environ.get("TRAJECTORY_DUMP_PREFIX", "trajectory"),
            )

        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )

        # Pause GPU filler — returning to fit() for GPU training (compute_log_prob, FSDP update)
        try:
            from scripts.gpu_filler_control import pause as gpu_filler_pause
            gpu_filler_pause()
        except Exception:
            pass

        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)
        
        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        if effective_rollout_n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % effective_rollout_n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = preprocess_batch(gen_batch=gen_batch, 
                                     obs=obs, 
                                     config=self.config, 
                                     tokenizer=self.tokenizer, 
                                     processor=self.processor,
                                     )

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            effective_rollout_n: int,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        is_dapo = str(self.config.algorithm.adv_estimator) in ("dapo", "AdvantageEstimator.DAPO")
        max_try_count = (
            self.config.algorithm.dapo.max_gen_batches
            if is_dapo
            else self.config.algorithm.filter_groups.max_num_gen_batches
        )
        extra_rollout_seconds = 0.0
        generated_prompt_groups = accepted_prompt_groups = 0
        rejected_all_zero = rejected_all_one = 0

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1
            round_start = time.monotonic()

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                # vanilla_multi_turn_loop consumes env_kwargs; each dynamic round
                # needs an intact prompt group and environment payload.
                gen_batch=deepcopy(gen_batch),
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
            )
            if try_count > 1:
                extra_rollout_seconds += time.monotonic() - round_start
            if is_dapo:
                generated_prompt_groups += len(batch_list) // effective_rollout_n
                correctness = success.get("success_rate", episode_rewards)
                for group_start in range(0, len(batch_list), effective_rollout_n):
                    group_rewards = np.asarray(correctness[group_start:group_start + effective_rollout_n])
                    binary = group_rewards > 0
                    if binary.all():
                        rejected_all_one += 1
                    elif not binary.any():
                        rejected_all_zero += 1
                    else:
                        accepted_prompt_groups += 1
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        if is_dapo:
            from verl.trainer.ppo.core_algos import validate_dapo_dynamic_sampling_result

            if isinstance(actor_rollout_wg, dict):
                rollout_world_size = int(np.lcm.reduce([worker_group.world_size for worker_group in actor_rollout_wg.values()]))
            else:
                rollout_world_size = actor_rollout_wg.world_size
            validate_dapo_dynamic_sampling_result(
                collected_trajectories=len(total_batch_list),
                world_size=rollout_world_size,
                generation_rounds=try_count,
                max_generation_rounds=max_try_count,
            )
            self._dapo_dynamic_sampling_metrics = {
                "dapo/generated_prompt_groups": generated_prompt_groups,
                "dapo/accepted_prompt_groups": accepted_prompt_groups,
                "dapo/rejected_all_zero": rejected_all_zero,
                "dapo/rejected_all_one": rejected_all_one,
                "dapo/dynamic_sampling_acceptance_rate": accepted_prompt_groups / max(generated_prompt_groups, 1),
                "dapo/generation_rounds": try_count,
                "dapo/extra_rollout_seconds": extra_rollout_seconds,
            }

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        if is_dapo:
            target_trajectories = self.config.data.train_batch_size * effective_rollout_n
            if len(total_batch_list) > target_trajectories:
                total_batch_list = total_batch_list[:target_trajectories]
                total_episode_rewards = total_episode_rewards[:target_trajectories]
                total_episode_lengths = total_episode_lengths[:target_trajectories]
                total_success = {key: values[:target_trajectories] for key, values in total_success.items()}
                total_traj_uid = total_traj_uid[:target_trajectories]
                total_tool_callings = total_tool_callings[:target_trajectories]

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Determine the effective rollout_n based on training/validation mode
        if is_train:
            effective_rollout_n = self.config.env.rollout.n
            gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
        else:
            # For validation, use val_n for pass@k and avg@k computation
            val_rollout_n = getattr(self.config.env.rollout, 'val_n', None)
            if val_rollout_n is not None and val_rollout_n > 1:
                effective_rollout_n = val_rollout_n
                gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
            else:
                effective_rollout_n = 1
        gen_batch.meta_info["is_train"] = is_train
        self._current_global_step = int(gen_batch.meta_info.get("global_step", 0))

        # Capture env_kwargs before vanilla_multi_turn_loop pops them (for PW_DEBUG JSON dump)
        _pw_gt = None
        _pw_qs = None
        if os.environ.get("PW_DEBUG") == "True":
            ek = gen_batch.non_tensor_batch.get("env_kwargs", None)
            if ek is not None:
                _pw_gt = [kw.get("ground_truth", "") if kw else "" for kw in ek]
                _pw_qs = [kw.get("question", "") if kw else "" for kw in ek]

        from verl.trainer.ppo.core_algos import dapo_dynamic_sampling_enabled

        dapo_dynamic_sampling = dapo_dynamic_sampling_enabled(
            adv_estimator=self.config.algorithm.adv_estimator,
            dynamic_sampling=self.config.algorithm.dapo.dynamic_sampling,
            is_train=is_train,
        )
        if (self.config.algorithm.filter_groups.enable and is_train) or dapo_dynamic_sampling:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
            ground_truths=_pw_gt,
            questions=_pw_qs,
            is_train=is_train,
        )
        if dapo_dynamic_sampling:
            gen_batch_output.meta_info["dapo_dynamic_sampling_metrics"] = self._dapo_dynamic_sampling_metrics
        
        return gen_batch_output

# =============================================================================
# Multi‑Agent collector orchestrating a *team* of agents
# =============================================================================
class MultiAgentTrajectoryCollector(TrajectoryCollector):
    """Trajectory collector that *delegates* action generation to a
    user‑configurable :class:`BaseOrchestra` (chain, hierarchy, etc.)."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Any,
        wg_to_agents_mapping: Dict[str, List[Dict[str, str]]],
        tokenizers: Dict[str, PreTrainedTokenizer],
        processors: Dict[str, Any] = None,
    ):
        super().__init__(config=config, tokenizer=tokenizers, processor=processors)

        agent_ids = config.agent.agent_ids
        model_ids = config.agent.model_ids
        orchestra_type = config.agent.orchestra_type
        print("agent_ids: ", agent_ids)
        print("orchestra_type: ", orchestra_type)

        agents_to_wg_mapping = {}
        for wg_id, agents in wg_to_agents_mapping.items():
            for a in agents:
                agent_id = a['agent_id']
                agents_to_wg_mapping[agent_id] = wg_id

        if orchestra_type == "search":
            from agent_system.agent.orchestra.search import SearchMultiAgentOrchestra as orchestra
        elif orchestra_type == "math":
            from agent_system.agent.orchestra.math import MathMultiAgentOrchestra as orchestra
        else:
            raise ValueError(f"Unknown orchestra_type '{orchestra_type}'.")

        self.multiagent_orchestra: BaseOrchestra = orchestra(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )

        # Initialize FactExtractor / FactFilter for OPD 改进3 (fact-level privileged info)
        self._fact_extractor = None
        self._fact_filter = None
        self._fact_card_pipeline_enabled = False
        use_opd = str(getattr(config.algorithm, "use_opd", False)).strip().lower() in ("true", "1")
        use_facts = str(getattr(config.algorithm.opd, "use_facts", False)).strip().lower() in ("true", "1")
        fact_cfg = config.algorithm.opd.facts
        selection_mode = getattr(fact_cfg, "selection_mode", "legacy")
        if selection_mode not in {"legacy", "fact_card_loo"}:
            raise ValueError(f"Unknown algorithm.opd.facts.selection_mode={selection_mode!r}")
        estimator = str(config.algorithm.adv_estimator)
        if estimator == "rlsd" and not (use_opd and use_facts and selection_mode == "fact_card_loo"):
            raise ValueError(
                "algorithm.adv_estimator=rlsd requires algorithm.use_opd=True, "
                "algorithm.opd.use_facts=True, and facts.selection_mode=fact_card_loo"
            )
        if use_opd and use_facts and selection_mode == "fact_card_loo":
            if estimator not in {"opd_grpo_joint_loss", "rlsd"}:
                raise ValueError(
                    "algorithm.opd.facts.selection_mode=fact_card_loo requires "
                    "algorithm.adv_estimator=opd_grpo_joint_loss or rlsd"
                )
            self._fact_card_pipeline_enabled = True
            print(f"[OPD FactCard] enabled: model={fact_cfg.model_name}")
        elif use_opd and use_facts and selection_mode == "legacy":
            try:
                from verl.trainer.ppo.opd_facts import FactExtractor, FactFilter
                self._fact_extractor = FactExtractor(fact_cfg)
                self._fact_filter = FactFilter(fact_cfg)
                print(f"[OPD Facts] initialized: model={fact_cfg.model_name}")
            except Exception as e:
                print(f"[OPD Facts] Failed to initialize: {e}")

    # ------------------------------------------------------------------
    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        effective_rollout_n: int,
    ):

        batch_size = len(gen_batch.batch)

        # Extract ground_truth and question before env_kwargs is popped (for Judger agent)
        env_kwargs_list = gen_batch.non_tensor_batch.get('env_kwargs', None)
        ground_truths = None
        questions = None
        if env_kwargs_list is not None and gen_batch.meta_info.get("is_train", True):
            ground_truths = [kw.get('ground_truth', '') if kw else '' for kw in env_kwargs_list]
            questions = [kw.get('question', '') if kw else '' for kw in env_kwargs_list]

        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        self.multiagent_orchestra.reset()
        
        if effective_rollout_n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % effective_rollout_n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object) # grpo中的group-id
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object) # traj的唯一id
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            ###############################
            text_actions, multiagent_batch_buffer = self.multiagent_orchestra.run(
                gen_batch=gen_batch,
                env_obs=obs,
                actor_rollout_wgs=actor_rollout_wg,
                active_masks=active_masks,
                step=_step+1,
                ground_truths=ground_truths,
                questions=questions,
            )
            next_obs, rewards, dones, infos = envs.step(text_actions)
            ###############################
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"

            for data in multiagent_batch_buffer:
                agent_id, agent_batch = data['agent_id'], data['batch']
                agent_batch.non_tensor_batch['agent_id'] = np.array([agent_id for _ in range(batch_size)], dtype=object)
                agent_batch.non_tensor_batch['uid'] = uid_batch
                agent_batch.non_tensor_batch['traj_uid'] = traj_uid
                agent_batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
                # External agents (e.g., Judger) may already set active_masks
                # (False = don't enter effective_batch). Don't overwrite.
                if 'active_masks' not in agent_batch.non_tensor_batch:
                    agent_batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
                # OPD: store decoded text responses for teacher prompt construction
                wg_id = self.multiagent_orchestra.agents_to_wg_mapping.get(agent_id)
                if wg_id is not None:
                    tokenizer = self.tokenizer[wg_id]
                    text_responses_decoded = tokenizer.batch_decode(agent_batch.batch['responses'], skip_special_tokens=True)
                    agent_batch.non_tensor_batch['text_response'] = np.array(text_responses_decoded, dtype=object)
                raw_step_indices = np.array(
                    [len(total_batch_list[i]) for i in range(batch_size)], dtype=np.int64
                )
                global_step = int(gen_batch.meta_info.get("global_step", 0))
                agent_batch.non_tensor_batch["response_uid"] = np.array([
                    _stable_response_uid(global_step, traj_uid[i], raw_step_indices[i], agent_id)
                    for i in range(batch_size)
                ], dtype=object)
                agent_batch.non_tensor_batch["sequence_index"] = raw_step_indices
                agent_batch.non_tensor_batch["raw_step_index"] = raw_step_indices
                agent_batch.non_tensor_batch["agent_turn_index"] = np.array([
                    sum(step.get("agent_id") == agent_id for step in total_batch_list[i])
                    for i in range(batch_size)
                ], dtype=np.int64)
                agent_batch.non_tensor_batch["query_text"] = np.array([
                    questions[i] if questions is not None else "" for i in range(batch_size)
                ], dtype=object)
                # External agents (e.g., Judger) already have text_response set in non_tensor_batch
                agent_batch_list: list[dict] = to_list_of_dict(agent_batch)
                for i in range(batch_size):
                    if agent_batch_list[i]['agent_active_mask']:
                        total_batch_list[i].append(agent_batch_list[i])
                        total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
