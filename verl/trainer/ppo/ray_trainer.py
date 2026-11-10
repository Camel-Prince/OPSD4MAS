# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type
import pdb
import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.fact_pipeline_filler import fact_card_cpu_filler, fact_card_gpu_section
from verl.trainer.ppo.metric_utils import (
    advantage_metric_namespace,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    compute_pass_at_k_and_avg_at_k,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch, split_batch_by_wg_ids, combine_batches
from agent_system.agent.utils import build_wg_ids, normalize_agent_id, normalize_model_id

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    GSPO = "gspo"
    DAPO = "dapo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'
    OPD = "opd"
    OPD_GRPO = "opd_grpo"
    OPD_GRPO_JOINT = "opd_grpo_joint"                 # deprecated alias → OPD_GRPO_JOINT_ADV
    OPD_GRPO_JOINT_ADV = "opd_grpo_joint_adv"         # advantage-level joint (OPD+GRPO in one advantages tensor)
    OPD_GRPO_JOINT_LOSS = "opd_grpo_joint_loss"       # loss-level joint (two separate loss terms in actor)
    RLSD = "rlsd"                                      # fact-selected multiplicative GRPO credit
    OPD_LAST_OUTCOME = "opd_last_outcome"
    OPD_HYBRID = "opd_hybrid"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, group_by_agent_id=False, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # Determine grouping strategy based on configuration
    if group_by_agent_id:
        # Group by both uid and agent_id
        group_index = np.array([f"{uid}_{agent_id}" for uid, agent_id in zip(data.non_tensor_batch["uid"], data.non_tensor_batch["agent_id"])], dtype=object)
    else:
        # Use original uid grouping
        group_index = data.non_tensor_batch["uid"]
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator in (AdvantageEstimator.GSPO, AdvantageEstimator.DAPO):
        # Both algorithms retain GRPO's role-conditioned group advantage. Their
        # defining changes live in the actor loss and, for DAPO, rollout sampling.
        calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = calculation_mask.size(1)
            calculation_mask = data.batch["loss_mask"][:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=calculation_mask,
            index=group_index,
            traj_index=data.non_tensor_batch["traj_uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if adv_estimator == AdvantageEstimator.DAPO:
            masked_advantages = advantages[calculation_mask.bool()]
            data.meta_info["advantage_metrics"] = {
                "dapo/nonzero_adv_fraction": (
                    (masked_advantages != 0).float().mean().item() if masked_advantages.numel() else 0.0
                )
            }
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            group_by_agent_id=group_by_agent_id,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = calculation_mask.size(1)
            calculation_mask = data.batch["loss_mask"][:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=calculation_mask,
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        valid = calculation_mask.bool()
        valid_returns = returns[valid]
        valid_advantages = advantages[valid]
        data.meta_info["advantage_metrics"] = {
            "reinforce_plus_plus/return_mean": valid_returns.mean().item() if valid_returns.numel() else 0.0,
            "reinforce_plus_plus/return_std": valid_returns.std(unbiased=False).item() if valid_returns.numel() else 0.0,
            "reinforce_plus_plus/advantage_mean": valid_advantages.mean().item() if valid_advantages.numel() else 0.0,
            "reinforce_plus_plus/advantage_std": valid_advantages.std(unbiased=False).item() if valid_advantages.numel() else 0.0,
            "reinforce_plus_plus/nonzero_adv_fraction": (
                (valid_advantages != 0).float().mean().item() if valid_advantages.numel() else 0.0
            ),
        }
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = calculation_mask.size(1)
            calculation_mask = data.batch["loss_mask"][:, -response_length:]
        rloo_group_index = np.array(
            [f"{uid}_{agent_id}" for uid, agent_id in zip(data.non_tensor_batch["uid"], data.non_tensor_batch["agent_id"])],
            dtype=object,
        )
        advantages, returns, rloo_metrics = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=calculation_mask,
            index=rloo_group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            group_by_agent_id=group_by_agent_id,
            return_metrics=True,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["advantage_metrics"] = rloo_metrics
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_algos.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            group_by_agent_id=group_by_agent_id,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator in (AdvantageEstimator.OPD, AdvantageEstimator.OPD_GRPO):
        combine_with_grpo = (adv_estimator == AdvantageEstimator.OPD_GRPO)
        grpo_advantages = None
        if combine_with_grpo:
            grpo_calculation_mask = data.batch["response_mask"]
            if multi_turn:
                response_length = grpo_calculation_mask.size(1)
                grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
            grpo_advantages, _ = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=group_index,
                traj_index=data.non_tensor_batch['traj_uid'],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                group_by_agent_id=group_by_agent_id,
            )
        advantages, returns = core_algos.compute_opd_advantage(
            student_log_probs=data.batch["old_log_probs"],
            teacher_log_probs=data.batch["teacher_log_probs"],
            response_mask=data.batch["response_mask"],
            opd_lambda=kwargs.get("opd_lambda", 1.0),
            combine_with_grpo=combine_with_grpo,
            grpo_advantages=grpo_advantages,
            grpo_weight=kwargs.get("opd_grpo_weight", 0.5),
            power_opd_enabled=kwargs.get("opd_power_opd_enabled", False),
            power_alpha=kwargs.get("opd_power_alpha", 1.0),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator in (AdvantageEstimator.OPD_GRPO_JOINT_ADV, AdvantageEstimator.OPD_GRPO_JOINT):
        # advantage-level joint: OPD + GRPO combined into ONE advantages tensor
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        advantages, returns = core_algos.compute_opd_grpo_joint_advantage(
            student_log_probs=data.batch["old_log_probs"],
            teacher_log_probs=data.batch["teacher_log_probs"],
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            grpo_response_mask=grpo_calculation_mask,
            opd_beta=kwargs.get("opd_beta", 5e-3),
            opd_lambda=kwargs.get("opd_lambda", 1.0),
            grpo_outcome_weight=kwargs.get("opd_grpo_outcome_weight", 1.0),
            index=group_index,
            traj_index=data.non_tensor_batch["traj_uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
            power_opd_enabled=kwargs.get("opd_power_opd_enabled", False),
            power_alpha=kwargs.get("opd_power_alpha", 1.0),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPD_GRPO_JOINT_LOSS:
        # loss-level joint: OPD + GRPO as TWO separate advantages
        # (combined in actor as two independent loss terms)
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        adv_opd, adv_grpo = core_algos.compute_opd_grpo_joint_loss_advantages(
            student_log_probs=data.batch["old_log_probs"],
            teacher_log_probs=data.batch["teacher_log_probs"],
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            grpo_response_mask=grpo_calculation_mask,
            opd_lambda=kwargs.get("opd_lambda", 1.0),
            index=group_index,
            traj_index=data.non_tensor_batch["traj_uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
            power_opd_enabled=kwargs.get("opd_power_opd_enabled", False),
            power_alpha=kwargs.get("opd_power_alpha", 1.0),
        )
        data.batch["advantages_opd"] = adv_opd
        data.batch["advantages_grpo"] = adv_grpo
        data.batch["advantages"] = adv_grpo        # compat: downstream logging reads "advantages"
        data.batch["returns"] = adv_grpo
    elif adv_estimator == AdvantageEstimator.OPD_LAST_OUTCOME:
        is_last_step = torch.tensor(
            data.non_tensor_batch.get('is_last_step', np.zeros(len(data.batch["response_mask"]), dtype=bool)),
            dtype=torch.bool,
        )
        advantages, returns = core_algos.compute_opd_last_outcome_advantage(
            student_log_probs=data.batch["old_log_probs"],
            teacher_log_probs=data.batch["teacher_log_probs"],
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            is_last_step=is_last_step,
            opd_lambda=kwargs.get("opd_lambda", 1.0),
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
            power_opd_enabled=kwargs.get("opd_power_opd_enabled", False),
            power_alpha=kwargs.get("opd_power_alpha", 1.0),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPD_HYBRID:
        advantages, returns = core_algos.compute_opd_hybrid_advantage(
            student_log_probs=data.batch["old_log_probs"],
            teacher_log_probs=data.batch["teacher_log_probs"],
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            opd_lambda=kwargs.get("opd_lambda", 1.0),
            outcome_weight=kwargs.get("opd_outcome_weight", 0.3),
            index=group_index,
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
            power_opd_enabled=kwargs.get("opd_power_opd_enabled", False),
            power_alpha=kwargs.get("opd_power_alpha", 1.0),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizers,
        default_wg,
        wg_to_agents_mapping: dict[str, list[dict[str, str]]],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processors=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizers = tokenizers
        self.processors = processors
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        # if self.hybrid_engine:
        #     assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.multi_agent = self.config.agent.multi_agent
        assert self.multi_agent
        self.agent_ids = self.config.agent.agent_ids
        self.model_ids = self.config.agent.model_ids
        assert len(self.agent_ids) == len(self.model_ids), "agent_ids and model_ids must have the same length"

        self.model_sharing = self.config.agent.model_sharing
        self.wg_to_agents_mapping = wg_to_agents_mapping

        self.default_wg = default_wg

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GSPO,
            AdvantageEstimator.DAPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO,
            AdvantageEstimator.OPD,
            AdvantageEstimator.OPD_GRPO,
            AdvantageEstimator.OPD_GRPO_JOINT,
            AdvantageEstimator.OPD_GRPO_JOINT_ADV,
            AdvantageEstimator.OPD_GRPO_JOINT_LOSS,
            AdvantageEstimator.RLSD,
            AdvantageEstimator.OPD_LAST_OUTCOME,
            AdvantageEstimator.OPD_HYBRID,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_multi_agent_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
    

    def _validate_multi_agent_config(self):
        """Validate configuration for each agent after agent-specific parameters are applied."""
        from omegaconf import OmegaConf
        
        for wg_id, agents_configs in self.wg_to_agents_mapping.items():
            for ac in agents_configs:
                agent_id = ac.get("agent_id", "unknown")
                # Create a temporary config by merging agent-specific actor_rollout_ref with base config
                temp_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
                # Replace actor_rollout_ref with agent-specific config
                temp_config.actor_rollout_ref = ac["config_actor_rollout_ref"]
                # Validate this agent's config
                try:
                    self._validate_config(temp_config)
                except (ValueError, AssertionError) as e:
                    raise ValueError(
                        f"[Agent {agent_id} in WG {wg_id}] Configuration validation failed: {str(e)}"
                    ) from e

    def _validate_config(self, config):
        # config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        actor_config = config.actor_rollout_ref.actor
        for mode_key in ("loss_agg_mode", "grpo_loss_agg_mode", "opd_loss_agg_mode"):
            mode = actor_config.get(mode_key, actor_config.loss_agg_mode)
            assert mode in valid_loss_agg_modes, f"Invalid {mode_key}: {mode}"

        if config.algorithm.adv_estimator == AdvantageEstimator.GSPO:
            strategy = actor_config.get("strategy", "fsdp")
            if strategy not in ("fsdp", "fsdp2"):
                raise NotImplementedError(
                    f"GSPO sequence-level clipping is implemented only for the data-parallel FSDP actor; got {strategy!r}."
                )
        if config.algorithm.adv_estimator == AdvantageEstimator.DAPO:
            dapo = config.algorithm.dapo
            if dapo.loss_agg_mode != "token-mean":
                raise ValueError("DAPO requires algorithm.dapo.loss_agg_mode=token-mean")
            if dapo.overlong_cache_len < 0 or dapo.overlong_cache_len >= config.data.max_response_length:
                raise ValueError(
                    "algorithm.dapo.overlong_cache_len must be 0 or satisfy "
                    "0 < overlong_cache_len < data.max_response_length"
                )
            if dapo.dynamic_sampling and dapo.max_gen_batches <= 0:
                raise ValueError("algorithm.dapo.max_gen_batches must be positive when dynamic sampling is enabled")
            if dapo.dynamic_sampling and config.env.rollout.n < 2:
                raise ValueError("DAPO dynamic sampling requires env.rollout.n >= 2")

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizers[self.default_wg], self.processors[self.default_wg])
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizers[self.default_wg], self.processors[self.default_wg])
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        uid_list = []  # Add uid collection for grouping
        task_pass_lst = []
        # success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # Add progress bar for validation
        val_progress_bar = tqdm(total=len(self.val_dataloader), desc="Validation Progress")

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                val_progress_bar.close()
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizers[self.default_wg].decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizers[self.default_wg].eos_token_id,
                "pad_token_id": self.tokenizers[self.default_wg].pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            if self.config.algorithm.adv_estimator == AdvantageEstimator.DAPO:
                try:
                    from scripts.gpu_filler_control import resume as gpu_filler_resume
                    gpu_filler_resume(reason="DAPO validation reward/decode phase")
                except Exception:
                    pass
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            wg_ids = test_output_gen_batch.non_tensor_batch["wg_id"]
            output_texts = [self.tokenizers[model].decode(ids, skip_special_tokens=True) for ids, model in zip(output_ids, wg_ids)]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            validation_output = self.config.trainer.get("validation_data_dir", None)
            if validation_output:
                from scripts.evaluation_records import append_validation_records
                append_validation_records(validation_output, test_batch, input_texts, output_texts, scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            uid_list.append(test_output_gen_batch.non_tensor_batch['uid'])  # Collect uid for grouping
            task_pass_lst.append(test_output_gen_batch.non_tensor_batch['pass'])

            # Update validation progress bar
            val_progress_bar.update(1)

            if os.environ.get("PW_DEBUG") == "True":
                break  # only validate first batch in debug mode

            # success rate
            # for k in test_batch.non_tensor_batch.keys():
            #     if 'success_rate' in k:
            #         if k not in success_rate_dict:
            #             success_rate_dict[k] = []
            #         success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
            #         # all success_rate should be the same
            #         for i in range(1, len(test_batch.non_tensor_batch[k])):
            #             assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        uids = np.concatenate(uid_list, axis=0)  # Group UIDs for task grouping
        task_pass = np.concatenate(task_pass_lst, axis=0)
        # success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # Get validation rollout n for pass@k and avg@k computation
        val_rollout_n = getattr(self.config.env.rollout, 'val_n', 1)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]
        unique_uids = uids[unique_idx]
        unique_task_pass = task_pass[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        # for k, v in success_rate.items():
        #     metric_dict[f'val/{k.replace("success_rate", "pass@1")}'] = v
        
        # Compute pass@k and avg@k for all data
        passk_avgk_metrics = compute_pass_at_k_and_avg_at_k(
            unique_uids=unique_uids,
            unique_task_pass=unique_task_pass,
            k=val_rollout_n,
            strict_group_size=True,
        )
        metric_dict.update({f'val/{k}': v for k, v in passk_avgk_metrics.items()})
        
        # Compute pass@k and avg@k per data source
        for data_source in set(unique_data_sources):
            data_source_mask = unique_data_sources == data_source
            if np.any(data_source_mask):
                ds_passk_avgk_metrics = compute_pass_at_k_and_avg_at_k(
                    unique_uids=unique_uids[data_source_mask],
                    unique_task_pass=unique_task_pass[data_source_mask],
                    k=val_rollout_n,
                    strict_group_size=True,
                )
                metric_dict.update({f'val/{data_source}/{k}': v for k, v in ds_passk_avgk_metrics.items()})

        val_progress_bar.close()
        if self.config.algorithm.adv_estimator == AdvantageEstimator.DAPO:
            try:
                from scripts.gpu_filler_control import pause as gpu_filler_pause
                gpu_filler_pause(reason="after DAPO validation")
            except Exception:
                pass
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Set configurations for each model
        wg_configs = {}
        for wg_id, agents_list in self.wg_to_agents_mapping.items():
            wg_configs[wg_id] = agents_list[0]["config_actor_rollout_ref"]
            wg_configs[wg_id].model.path = agents_list[0]["model_id"]

        # create actor and rollout
        if self.hybrid_engine:
            for wg_id, agents_list in self.wg_to_agents_mapping.items():
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                role_name = "actor_rollout/" + "+".join(normalize_agent_id(agent_model["agent_id"]) for agent_model in agents_list) + "/"
                actor_rollout_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ActorRollout],
                    config=wg_configs[wg_id],
                    role=role_name,
                )
                self.resource_pool_to_cls[resource_pool][role_name] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            for wg_id, agents_list in self.wg_to_agents_mapping.items():
                role_name = "ref/" + "+".join(normalize_agent_id(agent_model["agent_id"]) for agent_model in agents_list) + "/"
                # we use the same config for all ref policies
                ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=wg_configs[wg_id], role=role_name)
                self.resource_pool_to_cls[resource_pool][role_name] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = {}
            for wg_id, agents_list in self.wg_to_agents_mapping.items():
                role_name = "ref/" + "+".join(normalize_agent_id(agent_model["agent_id"]) for agent_model in agents_list) + "/"
                assert role_name in all_wg, f"Role {role_name} not found in all_wg"
                ref_policy_wg = all_wg[role_name]
                ref_policy_wg.init_model()
                self.ref_policy_wg[wg_id] = ref_policy_wg

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = {}
        for wg_id, agents_list in self.wg_to_agents_mapping.items():
            role_name = "actor_rollout/" + "+".join(normalize_agent_id(agent_model["agent_id"]) for agent_model in agents_list) + "/"
            assert role_name in all_wg, f"Role {role_name} not found in all_wg"
            actor_rollout_wg = all_wg[role_name]
            actor_rollout_wg.init_model()

            self.actor_rollout_wg[wg_id] = actor_rollout_wg

    def _save_checkpoint(self):
        # import pdb
        # pdb.set_trace()
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        import sys
        print(f"local_global_step_folder: {local_global_step_folder}", file=sys.stderr, flush=True)

        for wg_id in self.wg_to_agents_mapping.keys():
            actor_local_path = os.path.join(local_global_step_folder, f"actor/{wg_id}")

            actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", f"actor/{wg_id}")

            remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
            if remove_previous_ckpt_in_save:
                print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
            max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
            max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

            self.actor_rollout_wg[wg_id].save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        # load actor
        for wg_id in self.wg_to_agents_mapping.keys():
            actor_path = os.path.join(global_step_folder, f"actor/{wg_id}")
            if not os.path.isdir(actor_path):
                from pathlib import Path
                role = self.wg_to_agents_mapping[wg_id][0]["agent_id"]
                matches = list((Path(global_step_folder) / "actor").glob(f"*_{role}"))
                if len(matches) != 1:
                    raise ValueError(f"Cannot uniquely map checkpoint role {role}")
                actor_path = str(matches[0])
            self.actor_rollout_wg[wg_id].load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        critic_path = os.path.join(global_step_folder, "critic")
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if not self.config.trainer.get("val_only", False) and os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, wg_id, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg[wg_id].world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _build_teacher_batch(self, sub_batch: DataProto, wg_id: str, use_response_mask: bool = False) -> DataProto:
        """Build a teacher batch by replacing student prompts with teacher prompts (which include
        privileged future context), while keeping the same response tokens.

        The teacher forward pass uses torch.no_grad() inside compute_log_prob, providing
        the stop-gradient behavior needed for OPD.
        """
        import copy
        teacher_batch = copy.deepcopy(sub_batch)

        teacher_input_ids = sub_batch.batch["teacher_input_ids"]
        teacher_attention_mask = sub_batch.batch["teacher_attention_mask"]
        teacher_position_ids = sub_batch.batch["teacher_position_ids"]
        responses = sub_batch.batch["responses"]

        # Concatenate teacher prompt with original responses
        full_input_ids = torch.cat([teacher_input_ids, responses], dim=1)
        if use_response_mask:
            response_attn_mask = sub_batch.batch["response_mask"].to(teacher_attention_mask.dtype)
        else:
            response_attn_mask = torch.ones_like(responses, dtype=teacher_attention_mask.dtype)
        full_attention_mask = torch.cat([teacher_attention_mask, response_attn_mask], dim=1)

        # Recompute position_ids for the full sequence
        from verl.utils.model import compute_position_id_with_mask
        full_position_ids = compute_position_id_with_mask(full_attention_mask)

        teacher_batch.batch["input_ids"] = full_input_ids
        teacher_batch.batch["attention_mask"] = full_attention_mask
        teacher_batch.batch["position_ids"] = full_position_ids

        return teacher_batch

    def _fact_card_mode_enabled(self) -> bool:
        opd = self.config.algorithm.get("opd", {})
        facts = opd.get("facts", {})
        return str(self.config.algorithm.get("use_opd", False)).strip().lower() in {"true", "1"} and str(
            opd.get("use_facts", False)
        ).strip().lower() in {"true", "1"} and facts.get("selection_mode", "legacy") == "fact_card_loo"

    def _compute_fact_teacher_variants(self, batch: DataProto, variants):
        """Compute local-policy teacher log-probs for (response_uid, lessons) variants."""
        from verl.trainer.ppo.opd_teacher import build_fact_teacher_prompt, pad_teacher_data

        outputs = {}
        by_wg = defaultdict(list)
        response_to_wg = dict(zip(batch.non_tensor_batch["response_uid"], batch.non_tensor_batch["wg_id"]))
        for variant_key, response_uid, lessons in variants:
            wg_id = response_to_wg.get(response_uid)
            if wg_id is not None:
                by_wg[wg_id].append((variant_key, response_uid, lessons))
        facts_cfg = self.config.algorithm.opd.facts
        limit = max(1, int(facts_cfg.get("loo_variant_batch_size", 256)))
        max_context = int(self.config.algorithm.opd.get("max_teacher_context_length", 4096))
        for wg_id, wg_variants in by_wg.items():
            sub_batch = split_batch_by_wg_ids(batch, [wg_id])[wg_id]
            uid_to_index = {str(uid): index for index, uid in enumerate(sub_batch.non_tensor_batch["response_uid"])}
            tokenizer = self.tokenizers[wg_id]
            for start in range(0, len(wg_variants), limit):
                chunk = wg_variants[start:start + limit]
                row_indices = []
                teacher_items = []
                valid_keys = []
                for variant_key, response_uid, lessons in chunk:
                    row_index = uid_to_index.get(str(response_uid))
                    if row_index is None:
                        continue
                    response_length = int(sub_batch.batch["response_mask"][row_index].sum().item())
                    built = build_fact_teacher_prompt(
                        list(sub_batch.non_tensor_batch["student_prompt_messages"][row_index]),
                        list(lessons),
                        tokenizer,
                        response_length=response_length,
                        model_context_limit=max_context,
                        privileged_token_budget=int(facts_cfg.get("privileged_token_budget", 512)),
                    )
                    if not built["has_teacher"]:
                        continue
                    ids = torch.tensor(built["teacher_prompt_ids"], dtype=torch.long)
                    teacher_items.append({
                        "teacher_input_ids": ids,
                        "teacher_attention_mask": torch.ones_like(ids),
                        "teacher_position_ids": torch.arange(len(ids), dtype=torch.long),
                    })
                    row_indices.append(row_index)
                    valid_keys.append((variant_key, built))
                if not teacher_items:
                    continue
                padded = pad_teacher_data(
                    teacher_items,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                )
                variant_batch = sub_batch.select_idxs(row_indices)
                for key, value in padded.items():
                    variant_batch.batch[key] = value
                teacher_batch = self._build_teacher_batch(variant_batch, wg_id, use_response_mask=True)
                worker_group = self.actor_rollout_wg[wg_id]
                teacher_batch, pad_size = pad_dataproto_to_divisor(teacher_batch, worker_group.world_size)
                with fact_card_gpu_section("FactCard teacher log-prob"):
                    log_prob_output = worker_group.compute_log_prob(teacher_batch)
                log_prob_output = unpad_dataproto(log_prob_output, pad_size=pad_size)
                log_probs = log_prob_output.batch["old_log_probs"]
                for index, (variant_key, built) in enumerate(valid_keys):
                    outputs[variant_key] = (log_probs[index], built)
        return outputs

    def _run_fact_card_pipeline(self, batch: DataProto, metrics: Dict[str, float]) -> None:
        """Run external extraction/relation after async reward launch."""
        from verl.trainer.ppo.opd_fact_cards import (
            FactCardPipeline,
            ResponseRecord,
            should_save_trajectory_artifact,
        )

        records_payload = next((
            value for value in batch.non_tensor_batch.get("fact_records_payload", []) if value
        ), "")
        outcomes_payload = next((
            value for value in batch.non_tensor_batch.get("fact_outcomes_payload", []) if value
        ), "")
        trajectories_payload = next((
            value for value in batch.non_tensor_batch.get("fact_trajectories_payload", []) if value
        ), "")
        records = [ResponseRecord(**value) for value in json.loads(records_payload)] if records_payload else []
        outcomes = json.loads(outcomes_payload) if outcomes_payload else {"rewards": {}, "correctness": {}}
        if not hasattr(self, "_fact_card_pipeline"):
            self._fact_card_pipeline = FactCardPipeline(self.config.algorithm.opd.facts)
        cards, _, pipeline_metrics = self._fact_card_pipeline.run(
            records,
            outcomes["rewards"],
            outcomes["correctness"],
            self.global_steps,
        )
        if self.config.algorithm.opd.facts.get("save_trajectories", False) and trajectories_payload:
            trajectory_save_freq = int(
                self.config.algorithm.opd.facts.get("trajectory_save_freq", 1)
            )
            if should_save_trajectory_artifact(
                self.global_steps,
                self.total_training_steps,
                trajectory_save_freq,
            ):
                self._fact_card_pipeline.artifacts.write_trajectories(
                    self.global_steps, json.loads(trajectories_payload)
                )
        self._fact_card_pipeline.artifacts.flush()
        pipeline_metrics["facts/artifact_write_errors"] = len(
            self._fact_card_pipeline.artifacts.errors
        )
        batch.non_tensor_batch["fact_cards_payload"] = np.array(
            [json.dumps([card.to_dict() for card in cards], ensure_ascii=False)]
            + [""] * max(len(batch) - 1, 0), dtype=object
        )
        batch.non_tensor_batch["fact_metrics_payload"] = np.array(
            [json.dumps(pipeline_metrics, ensure_ascii=False)]
            + [""] * max(len(batch) - 1, 0), dtype=object
        )

    def _apply_fact_card_objective(self, batch: DataProto, metrics: Dict[str, float]) -> DataProto:
        """Run target LOO, selected-only teacher forward, and joint-loss/RLSD signal."""
        from verl.trainer.ppo.opd_fact_cards import (
            FactCard, SELECTED_DECISIONS, select_target_facts, selected_lessons_for_target,
        )

        payload = next((value for value in batch.non_tensor_batch.get("fact_cards_payload", []) if value), "")
        pipeline_metrics = next((value for value in batch.non_tensor_batch.get("fact_metrics_payload", []) if value), "")
        if pipeline_metrics:
            metrics.update(json.loads(pipeline_metrics))
        cards = [FactCard.from_dict(value) for value in json.loads(payload)] if payload else []
        cards_by_id = {card.fact_id: card for card in cards}
        uid_to_row = {str(uid): index for index, uid in enumerate(batch.non_tensor_batch["response_uid"])}
        candidates = defaultdict(list)
        for card in cards:
            for slot, target in enumerate(card.targets):
                if target.decision == "not_evaluated" and target.target_response_uid in uid_to_row:
                    candidates[target.target_response_uid].append((card.fact_id, slot))

        variants = []
        for target_uid, refs in candidates.items():
            lessons = [cards_by_id[fact_id].lesson_text for fact_id, _ in refs]
            variants.append(((target_uid, "full"), target_uid, lessons))
            for fact_id, _ in refs:
                variants.append(((target_uid, fact_id), target_uid, [
                    cards_by_id[other_id].lesson_text for other_id, _ in refs if other_id != fact_id
                ]))
        started = time.time()
        variant_outputs = self._compute_fact_teacher_variants(batch, variants)
        response_length = batch.batch["responses"].size(1)
        loss_mask = (
            batch.batch["loss_mask"][:, -response_length:]
            if "loss_mask" in batch.batch.keys()
            else batch.batch["response_mask"]
        )
        loo_values = []
        conflict_weight = float(self.config.algorithm.opd.facts.get("conflict_weight", 1.0))
        for target_uid, refs in candidates.items():
            row = uid_to_row[target_uid]
            full = variant_outputs.get((target_uid, "full"))
            for fact_id, slot in refs:
                without = variant_outputs.get((target_uid, fact_id))
                target = cards_by_id[fact_id].targets[slot]
                if without is None and len(refs) == 1:
                    without = (batch.batch["old_log_probs"][row], None)
                if full is None or without is None:
                    target.decision = "loo_rejected"
                    continue
                values = core_algos.compute_fact_loo_metrics(
                    full[0], without[0], batch.batch["advantages_grpo"][row], loss_mask[row], conflict_weight
                )
                target.loo_effect_rms = float(values["effect_rms"].item())
                target.loo_net_alignment = float(values["net_alignment"].item())
                target.loo_directional_precision = float(values["directional_precision"].item())
                loo_values.append(values)

        branch = "rlsd" if self.config.algorithm.adv_estimator == AdvantageEstimator.RLSD else "joint_loss"
        selected_by_target = {}
        facts_cfg = self.config.algorithm.opd.facts
        for target_uid, refs in candidates.items():
            selected_by_target[target_uid] = select_target_facts(
                cards_by_id, refs, branch,
                float(facts_cfg.min_effect), float(facts_cfg.min_precision),
                float(facts_cfg.min_net_alignment),
                int(facts_cfg.max_selected_joint_loss), int(facts_cfg.max_selected_rlsd_aligned),
                int(facts_cfg.max_selected_rlsd_uncertain), float(facts_cfg.uncertain_min_association),
            )
        selected_variants = []
        for target_uid, refs in selected_by_target.items():
            lessons = selected_lessons_for_target(cards_by_id, refs)
            if lessons:
                selected_variants.append(((target_uid, "selected"), target_uid, lessons))
        selected_outputs = self._compute_fact_teacher_variants(batch, selected_variants)
        teacher_log_probs = batch.batch["old_log_probs"].clone()
        has_teacher = torch.zeros(len(batch), dtype=torch.bool, device=teacher_log_probs.device)
        privileged_counts = []
        truncated = []
        for target_uid in selected_by_target:
            selected = selected_outputs.get((target_uid, "selected"))
            if selected is None:
                for fact_id, target_slot in selected_by_target[target_uid]:
                    cards_by_id[fact_id].targets[target_slot].decision = "loo_rejected"
                continue
            row = uid_to_row[target_uid]
            teacher_log_probs[row] = selected[0].to(teacher_log_probs.device)
            has_teacher[row] = bool(selected[1]["has_teacher"])
            privileged_counts.append(selected[1]["actual_privileged_tokens"])
            truncated.append(selected[1]["truncated"])
        batch.batch["teacher_log_probs"] = teacher_log_probs
        final_delta = (teacher_log_probs - batch.batch["old_log_probs"]) * loss_mask
        active_teacher_mask = loss_mask.bool() & has_teacher.unsqueeze(-1)
        metrics["opd/final_delta_rms"] = float(
            torch.sqrt(final_delta[active_teacher_mask].square().mean())
        ) if active_teacher_mask.any() else 0.0
        signed_delta = batch.batch["advantages_grpo"] * final_delta
        metrics["opd/grpo_token_sign_agreement"] = float(
            (signed_delta[active_teacher_mask] > 0).float().mean()
        ) if active_teacher_mask.any() else 0.0
        metrics["opd/grpo_token_conflict"] = float(
            (signed_delta[active_teacher_mask] < 0).float().mean()
        ) if active_teacher_mask.any() else 0.0
        metrics["facts/loo_seconds"] = time.time() - started
        metrics["facts/loo_candidates"] = sum(len(refs) for refs in candidates.values())
        metrics["facts/selected_aligned"] = sum(
            target.decision == "loo_aligned" for card in cards for target in card.targets
        )
        metrics["facts/selected_uncertain"] = sum(
            target.decision == "loo_uncertain" for card in cards for target in card.targets
        )
        metrics["facts/selected_direct"] = sum(
            target.future_distance == 1 and target.decision in SELECTED_DECISIONS
            for card in cards for target in card.targets
        )
        metrics["facts/selected_downstream"] = sum(
            target.future_distance > 1 and target.decision in SELECTED_DECISIONS
            for card in cards for target in card.targets
        )
        target_rows = sum(
            str(agent) != "Judger Agent" for agent in batch.non_tensor_batch["agent_id"]
        )
        metrics["facts/no_fact_target_fraction"] = 1.0 - len(selected_outputs) / max(target_rows, 1)
        for target_uid, refs in selected_by_target.items():
            if not refs or target_uid not in uid_to_row:
                continue
            row = uid_to_row[target_uid]
            agent_bucket = str(batch.non_tensor_batch["agent_id"][row]).replace(" ", "_")
            reward_bucket = "positive" if float(batch.batch["token_level_rewards"][row].sum()) > 0 else "nonpositive"
            target_adv = batch.batch["advantages_grpo"][row][loss_mask[row].bool()]
            advantage_bucket = "positive" if target_adv.numel() and float(target_adv.mean()) > 0 else "nonpositive"
            metrics[f"facts/selected_by_agent/{agent_bucket}"] = metrics.get(
                f"facts/selected_by_agent/{agent_bucket}", 0
            ) + len(refs)
            metrics[f"facts/selected_by_reward/{reward_bucket}"] = metrics.get(
                f"facts/selected_by_reward/{reward_bucket}", 0
            ) + len(refs)
            metrics[f"facts/selected_by_advantage/{advantage_bucket}"] = metrics.get(
                f"facts/selected_by_advantage/{advantage_bucket}", 0
            ) + len(refs)
        metrics["training/fact_teacher_actual_privileged_tokens_mean"] = float(np.mean(privileged_counts)) if privileged_counts else 0.0
        metrics["training/fact_teacher_truncated_fraction"] = float(np.mean(truncated)) if truncated else 0.0
        if loo_values:
            for metric_name in ("effect_rms", "net_alignment", "directional_precision"):
                metrics[f"facts/loo_{metric_name}_mean"] = float(torch.stack([
                    value[metric_name].detach().cpu() for value in loo_values
                ]).mean())

        if branch == "joint_loss":
            opd_cfg = self.config.algorithm.opd
            power_cfg = opd_cfg.get("power", {})
            batch.batch["advantages_opd"] = core_algos.compute_opd_distill_signal(
                teacher_log_probs,
                batch.batch["old_log_probs"],
                loss_mask,
                float(opd_cfg.get("lambda", 1.0)),
                power_opd_enabled=str(power_cfg.get("enabled", False)).strip().lower() in {"true", "1"},
                power_alpha=float(power_cfg.get("alpha", 1.0)),
            )
            batch.batch["advantages"] = batch.batch["advantages_grpo"]
            diagnostics_freq = int(facts_cfg.get("gradient_diagnostics_freq", 20))
            batch.meta_info["opd_gradient_diagnostics"] = diagnostics_freq > 0 and self.global_steps % diagnostics_freq == 0
        else:
            rlsd_cfg = self.config.algorithm.opd.rlsd
            alpha = core_algos.rlsd_alpha(
                max(self.global_steps - 1, 0),
                rlsd_cfg.alpha_init,
                rlsd_cfg.alpha_final,
                rlsd_cfg.anneal_steps,
            )
            advantages, credit, multiplier, log_credit = core_algos.compute_rlsd_advantage(
                batch.batch["advantages_grpo"], teacher_log_probs, batch.batch["old_log_probs"],
                loss_mask, has_teacher, float(rlsd_cfg.eps_w), alpha,
            )
            batch.batch["advantages"] = advantages
            batch.batch["returns"] = advantages
            batch.meta_info["rlsd"] = True
            active = loss_mask.bool()
            active_credit = credit[active]
            active_multiplier = multiplier[active]
            sorted_credit = torch.sort(active_credit).values if active_credit.numel() else active_credit
            low = 1.0 - float(rlsd_cfg.eps_w)
            high = 1.0 + float(rlsd_cfg.eps_w)
            metrics.update({
                "rlsd/alpha": alpha,
                "rlsd/credit_mean": float(active_credit.mean()) if active_credit.numel() else 1.0,
                "rlsd/credit_std": float(active_credit.std()) if active_credit.numel() > 1 else 0.0,
                "rlsd/multiplier_mean": float(active_multiplier.mean()) if active_multiplier.numel() else 1.0,
                "rlsd/multiplier_std": float(active_multiplier.std()) if active_multiplier.numel() > 1 else 0.0,
                "rlsd/credit_p05": float(torch.quantile(sorted_credit, 0.05)) if sorted_credit.numel() else 1.0,
                "rlsd/credit_p95": float(torch.quantile(sorted_credit, 0.95)) if sorted_credit.numel() else 1.0,
                "rlsd/credit_clip_low_fraction": float((active_credit <= low + 1e-7).float().mean()) if active_credit.numel() else 0.0,
                "rlsd/credit_clip_high_fraction": float((active_credit >= high - 1e-7).float().mean()) if active_credit.numel() else 0.0,
                "rlsd/sign_flip_fraction": float(((torch.sign(advantages) != torch.sign(batch.batch["advantages_grpo"])) & active).float().sum() / active.sum().clamp_min(1)),
                "rlsd/adv_rms_ratio": float(torch.sqrt((advantages[active].square()).mean()) / torch.sqrt((batch.batch["advantages_grpo"][active].square()).mean()).clamp_min(1e-8)) if active.any() else 1.0,
            })
        artifact_dir = facts_cfg.get("artifact_dir")
        final_save_freq = int(facts_cfg.get("final_save_freq", 20))
        if artifact_dir:
            from verl.trainer.ppo.opd_fact_cards import (
                FactArtifactStore,
                should_save_final_fact_artifact,
            )
            save_final = should_save_final_fact_artifact(
                self.global_steps,
                self.total_training_steps,
                final_save_freq,
            )
        else:
            save_final = False
        if save_final:
            if not hasattr(self, "_fact_artifact_store"):
                self._fact_artifact_store = FactArtifactStore(artifact_dir)
            self._fact_artifact_store.write_cards("final", self.global_steps, cards)
            self._fact_artifact_store.flush()
            metrics["facts/artifact_write_errors"] = len(self._fact_artifact_store.errors)
        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        gate_baselines_json = os.environ.get("INITIAL_VAL_GATE_BASELINES_JSON")
        logger = None

        def initialize_tracking():
            return Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        # Rejected gate attempts should not create/sync disposable W&B runs.
        # Normal runs retain the original eager logger initialization.
        if not gate_baselines_json:
            logger = initialize_tracking()

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            if gate_baselines_json:
                from verl.trainer.ppo.initial_val_gate import evaluate_initial_validation_gate

                gate_result = evaluate_initial_validation_gate(
                    val_metrics,
                    json.loads(gate_baselines_json),
                    float(os.environ.get("INITIAL_VAL_GATE_TOLERANCE", "0.01")),
                )
                gate_result["global_step"] = self.global_steps
                gate_result["env_seed"] = int(self.config.env.seed)
                gate_status_path = os.environ.get("INITIAL_VAL_GATE_STATUS_PATH")
                if gate_status_path:
                    gate_status_dir = os.path.dirname(os.path.abspath(gate_status_path))
                    os.makedirs(gate_status_dir, exist_ok=True)
                    gate_status_tmp = f"{gate_status_path}.tmp.{os.getpid()}"
                    with open(gate_status_tmp, "w", encoding="utf-8") as handle:
                        json.dump(gate_result, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    os.replace(gate_status_tmp, gate_status_path)
                for metric_name, comparison in gate_result["metrics"].items():
                    print(
                        f"[InitialValGate] {metric_name}: observed={comparison['observed']} "
                        f"minimum={comparison['minimum']:.6f} passed={comparison['passed']}",
                        flush=True,
                    )
                if not gate_result["passed"]:
                    raise RuntimeError("[InitialValGate] FAILED before training")
                print("[InitialValGate] PASSED; continuing training", flush=True)
            if logger is None:
                logger = initialize_tracking()
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if logger is None:
            logger = initialize_tracking()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()
                        ################ agent-environment loop ###############
                        gen_batch.meta_info["global_step"] = self.global_steps
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                        metrics.update(gen_batch_output.meta_info.get("dapo_dynamic_sampling_metrics", {}))
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_algos.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    # multiagent_batch (Dict[str, DataProto]): Dictionary mapping unique_wg_ids to their respective DataProto batches
                    unique_wg_ids = list(self.wg_to_agents_mapping.keys())

                    multiagent_batch: Dict[str, DataProto] = split_batch_by_wg_ids(batch, unique_wg_ids)

                    for wg_id in multiagent_batch.keys():
                        sub_batch = multiagent_batch[wg_id]
                        
                        # Create agent-specific config for adjust_batch
                        agent_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
                        # Replace actor_rollout_ref with agent-specific config
                        agent_config.actor_rollout_ref = self.wg_to_agents_mapping[wg_id][0]["config_actor_rollout_ref"]
                        
                        sub_batch = adjust_batch(agent_config, sub_batch, wg_id=wg_id)
                        multiagent_batch[wg_id] = sub_batch
                    
                        sub_batch.batch["response_mask"] = compute_response_mask(sub_batch)
                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        if self.config.trainer.balance_batch:
                            self._balance_batch(sub_batch, wg_id, metrics=metrics, logging_prefix= f"global_seqlen/{wg_id}")

                        # compute global_valid tokens
                        sub_batch.meta_info[f"{wg_id}/global_token_num"] = torch.sum(sub_batch.batch["attention_mask"], dim=-1).tolist()

                        multiagent_batch[wg_id] = sub_batch

                    batch: DataProto = combine_batches(multiagent_batch)

                    batch.meta_info[f"global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizers)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    if self._fact_card_mode_enabled():
                        with fact_card_cpu_filler("FactCard extraction/relation/association"):
                            with _timer("fact_pipeline", timing_raw):
                                self._run_fact_card_pipeline(batch, metrics)

                    multiagent_batch: Dict[str, DataProto] = split_batch_by_wg_ids(batch, unique_wg_ids)
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        for wg_id in multiagent_batch.keys():
                            sub_batch = multiagent_batch[wg_id]
                            assert all(wg_id == wg for wg in sub_batch.non_tensor_batch["wg_id"])
                            old_log_prob = self.actor_rollout_wg[wg_id].compute_log_prob(sub_batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = sub_batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {f"actor/{wg_id}/entropy_loss": entropy_loss.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            sub_batch = sub_batch.union(old_log_prob)

                            if "rollout_log_probs" in sub_batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = sub_batch.batch["rollout_log_probs"]
                                actor_old_log_probs = sub_batch.batch["old_log_probs"]
                                attention_mask = sub_batch.batch["attention_mask"]
                                responses = sub_batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        f"training/{wg_id}/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        f"training/{wg_id}/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        f"training/{wg_id}/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )
                            multiagent_batch[wg_id] = sub_batch

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            for wg_id in multiagent_batch.keys():
                                sub_batch = multiagent_batch[wg_id]
                                assert all(wg_id == wg for wg in sub_batch.non_tensor_batch["wg_id"])
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg[wg_id].compute_ref_log_prob(sub_batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg[wg_id].compute_ref_log_prob(sub_batch)
                                sub_batch = sub_batch.union(ref_log_prob)

                                multiagent_batch[wg_id] = sub_batch

                    # OPD: compute teacher log-probs with privileged future context
                    if self.config.algorithm.get("use_opd", False) and not self._fact_card_mode_enabled():
                        with _timer("teacher_log_prob", timing_raw):
                            for wg_id in multiagent_batch.keys():
                                sub_batch = multiagent_batch[wg_id]
                                if "teacher_input_ids" not in sub_batch.batch:
                                    continue
                                teacher_sub_batch = self._build_teacher_batch(sub_batch, wg_id)
                                teacher_log_prob_output = self.actor_rollout_wg[wg_id].compute_log_prob(teacher_sub_batch)
                                sub_batch.batch["teacher_log_probs"] = teacher_log_prob_output.batch["old_log_probs"]
                                multiagent_batch[wg_id] = sub_batch

                    batch: DataProto = combine_batches(multiagent_batch)
                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if (
                            self.config.algorithm.adv_estimator != AdvantageEstimator.DAPO
                            and self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True)
                        ):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.DAPO:
                            response_mask = batch.batch["response_mask"]
                            shaped_rewards, _, dapo_reward_metrics = core_algos.apply_dapo_overlong_reward_shaping(
                                token_level_rewards=batch.batch["token_level_rewards"],
                                response_mask=response_mask,
                                max_response_length=self.config.data.max_response_length,
                                cache_len=self.config.algorithm.dapo.overlong_cache_len,
                            )
                            batch.batch["token_level_rewards"] = shaped_rewards
                            metrics.update(dapo_reward_metrics)

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        group_by_agent_id = self.config.algorithm.get("group_by_agent_id", False) # 
                        opd_config = self.config.algorithm.get("opd", {})
                        opd_beta = max(
                            opd_config.get("beta_init", 5e-3)
                            - self.global_steps * opd_config.get("beta_anneal_rate", 2e-5),
                            opd_config.get("beta_min", 1e-3),
                        )
                        # loss-level joint annealing (independent of adv-level opd_beta)
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.OPD_GRPO_JOINT_LOSS:
                            opd_loss_weight = max(
                                opd_config.get("loss_weight_init", 0.1)
                                - self.global_steps * opd_config.get("loss_weight_anneal_rate", 4e-4),
                                opd_config.get("loss_weight_min", 0.02),
                            )
                            opd_loss_grpo_w = opd_config.get("loss_grpo_outcome_weight", 1.0)
                            batch.meta_info["opd_joint_loss"] = True
                            batch.meta_info["opd_loss_weight"] = opd_loss_weight
                            batch.meta_info["opd_loss_grpo_outcome_weight"] = opd_loss_grpo_w

                        fact_card_mode = self._fact_card_mode_enabled()
                        advantage_estimator = AdvantageEstimator.GRPO if fact_card_mode else self.config.algorithm.adv_estimator
                        batch = compute_advantage(
                            batch,
                            adv_estimator=advantage_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            group_by_agent_id=group_by_agent_id,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            opd_lambda=self.config.algorithm.get("opd", {}).get("lambda", 1.0),
                            opd_grpo_weight=self.config.algorithm.get("opd", {}).get("grpo_weight", 0.5),
                            opd_outcome_weight=self.config.algorithm.get("opd", {}).get("outcome_weight", 0.3),
                            opd_beta=opd_beta,
                            opd_grpo_outcome_weight=opd_config.get("grpo_outcome_weight", 1.0),
                            opd_power_opd_enabled=str(
                                opd_config.get("power", {}).get("enabled", False)
                            ).strip().lower() in {"true", "1"},
                            opd_power_alpha=float(opd_config.get("power", {}).get("alpha", 1.0)),
                        )
                        metrics.update(batch.meta_info.pop("advantage_metrics", {}))
                        if fact_card_mode:
                            batch.batch["advantages_grpo"] = batch.batch["advantages"].clone()
                            with fact_card_cpu_filler("FactCard LOO CPU preparation/selection"):
                                batch = self._apply_fact_card_objective(batch, metrics)

                        # OPD metrics logging
                        if self.config.algorithm.get("use_opd", False) and "teacher_log_probs" in batch.batch:
                            response_mask = batch.batch.get("response_mask", None)
                            if response_mask is not None:
                                adv_estimator = self.config.algorithm.adv_estimator
                                advantage_key = (
                                    "advantages_grpo"
                                    if adv_estimator == AdvantageEstimator.OPD_GRPO_JOINT_LOSS
                                    else "advantages"
                                )
                                advantage_namespace = advantage_metric_namespace(adv_estimator)
                                adv_vals = batch.batch[advantage_key]
                                masked_adv = torch.masked_select(adv_vals, response_mask.bool())
                                if masked_adv.numel() > 0:
                                    metrics.update({
                                        f"{advantage_namespace}/advantage_mean": masked_adv.mean().item(),
                                        f"{advantage_namespace}/advantage_std": masked_adv.std().item(),
                                        f"{advantage_namespace}/advantage_abs_mean": masked_adv.abs().mean().item(),
                                    })
                                teacher_lp = batch.batch["teacher_log_probs"]
                                student_lp = batch.batch["old_log_probs"]
                                kl_diff = torch.masked_select(student_lp - teacher_lp, response_mask.bool())
                                if kl_diff.numel() > 0:
                                    metrics.update({
                                        "opd/kl_diff_mean": kl_diff.mean().item(),
                                        "opd/kl_diff_std": kl_diff.std().item(),
                                    })
                                if self.config.algorithm.adv_estimator in (AdvantageEstimator.OPD_GRPO_JOINT, AdvantageEstimator.OPD_GRPO_JOINT_ADV):
                                    metrics.update({
                                        "opd/joint_adv_beta": opd_beta,
                                        "opd/joint_adv_grpo_outcome_weight": opd_config.get("grpo_outcome_weight", 1.0),
                                    })
                                if self.config.algorithm.adv_estimator == AdvantageEstimator.OPD_GRPO_JOINT_LOSS:
                                    metrics.update({
                                        "opd/joint_loss_opd_weight": opd_loss_weight,
                                        "opd/joint_loss_grpo_outcome_weight": opd_loss_grpo_w,
                                    })

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        multiagent_batch: Dict[str, DataProto] = split_batch_by_wg_ids(batch, unique_wg_ids)
                        # update actor
                        with _timer("update_actor", timing_raw):
                            for wg_id in multiagent_batch.keys():
                                sub_batch = multiagent_batch[wg_id]
                                assert all(wg_id == wg for wg in sub_batch.non_tensor_batch["wg_id"])
                                sub_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                configured_estimator = self.config.algorithm.adv_estimator
                                sub_batch.meta_info["rl_algorithm"] = str(
                                    getattr(configured_estimator, "value", configured_estimator)
                                )
                                if self.config.algorithm.adv_estimator == AdvantageEstimator.GSPO:
                                    sub_batch.meta_info["gspo_clip_ratio_low"] = self.config.algorithm.gspo.clip_ratio_low
                                    sub_batch.meta_info["gspo_clip_ratio_high"] = self.config.algorithm.gspo.clip_ratio_high
                                elif self.config.algorithm.adv_estimator == AdvantageEstimator.DAPO:
                                    sub_batch.meta_info["dapo_clip_ratio_low"] = self.config.algorithm.dapo.clip_ratio_low
                                    sub_batch.meta_info["dapo_clip_ratio_high"] = self.config.algorithm.dapo.clip_ratio_high
                                    sub_batch.meta_info["dapo_loss_agg_mode"] = self.config.algorithm.dapo.loss_agg_mode
                                actor_output = self.actor_rollout_wg[wg_id].update_actor(sub_batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info[f"{wg_id}/metrics"])
                                metrics.update(actor_output_metrics)
                                multiagent_batch[wg_id] = sub_batch

                        batch: DataProto = combine_batches(multiagent_batch)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs, outputs = [], []
                            multiagent_batch: Dict[str, DataProto] = split_batch_by_wg_ids(batch, unique_wg_ids)
                            for wg_id in multiagent_batch.keys():
                                sub_batch = multiagent_batch[wg_id]
                                sub_inputs = self.tokenizers[wg_id].batch_decode(sub_batch.batch["prompts"], skip_special_tokens=True)
                                sub_outputs = self.tokenizers[wg_id].batch_decode(sub_batch.batch["responses"], skip_special_tokens=True)
                                inputs += sub_inputs
                                outputs += sub_outputs
                                
                            batch: DataProto = combine_batches(multiagent_batch)

                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )
                    import sys
                    # validate
                    print(f'validate freq: {self.config.trainer.test_freq}', file=sys.stderr, flush=True)
                    os.environ.get('PW_DEBUG') and __import__('scripts.gpu_filler_control', fromlist=['pause']).pause()
                    os.environ.get('PW_DEBUG') and __import__('pdb').set_trace()
                    
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        import sys as _sys
                        print(f"\n[Validate] step {self.global_steps} — running validation...",
                              file=_sys.stderr, flush=True)
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        print(f"[Validate] step {self.global_steps} done!\nVal_Metrics: {val_metrics} ")
                    os.environ.get('PW_DEBUG') and __import__('scripts.gpu_filler_control', fromlist=['resume']).resume()

                    
                    print(f'save freq: {self.config.trainer.save_freq}', file=sys.stderr, flush=True)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        print(f"[Checkpoint] saving at step {self.global_steps}...")
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                        print(f"[Checkpoint] saved!")
                    os.environ.get('PW_DEBUG') and __import__('scripts.gpu_filler_control', fromlist=['resume']).pause()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, unique_wg_ids=unique_wg_ids, group_n=self.config.env.rollout.n, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # Explicit step progress for pipe/tee environments where tqdm may not render
                import sys as _sys
                _lr = metrics.get("training/actor/lr", 0)
                _reward = metrics.get("training/reward/mean", 0)
                print(f"[Training] step {self.global_steps}/{self.total_training_steps} "
                      f"reward={_reward:.4f} lr={_lr:.2e}",
                      file=_sys.stderr, flush=True)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
