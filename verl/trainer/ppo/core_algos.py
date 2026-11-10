# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict, Counter
import math

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any, Optional


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    env_step = batch.non_tensor_batch['env_step'].astype(np.int32)
    # returns_by_traj_ = {}
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        traj_env_step = env_step[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        first_of_group = np.r_[True, traj_env_step[1:] != traj_env_step[:-1]]
        step_starts = np.flatnonzero(first_of_group)
        step_ends = np.r_[step_starts[1:], len(traj_env_step)]

        step_rewards = traj_rewards[step_starts].astype(np.float32)

        # Calculate returns
        step_returns = np.zeros_like(step_rewards, dtype=np.float32)
        running_return = 0.0
        
        # Calculate returns from the end to the start
        for k in reversed(range(len(step_rewards))):
            running_return = step_rewards[k] + gamma * running_return
            step_returns[k] = running_return
        
        traj_returns = np.zeros_like(traj_rewards, dtype=np.float32)
        for sr, s, e in zip(step_returns, step_starts, step_ends):
            traj_returns[s:e] = sr
            
        # Store the results
        # returns_by_traj_[uid] = traj_returns
        returns_by_traj[uid] = (traj_indices, traj_returns)
    
    # Recombine the returns into the original batch order
    # all_returns_ = np.zeros_like(rewards)
    # for i, uid in enumerate(traj_uids):
    #     traj_indices = np.where(traj_uids == uid)[0]
    #     idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
    #     all_returns_[i] = returns_by_traj_[uid][idx_in_traj]

    all_returns = np.zeros_like(rewards, dtype=np.float32)
    for uid in unique_traj_uids:
        traj_indices, traj_returns = returns_by_traj[uid]
        all_returns[traj_indices] = traj_returns

    # assert (all_returns==all_returns_).all()
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   group_by_agent_id: bool = False
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std, group_by_agent_id)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        group_by_agent_id: bool = False,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard trajectory-level GRPO), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages



def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    print("group_by_agent_id: ", group_by_agent_id)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    traj_accumulator = defaultdict(list)
    traj2avg = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            traj_accumulator[(index[i], traj_index[i])].append(scores[i])
        
        for (idx, t_idx), reward_list in traj_accumulator.items():
            if group_by_agent_id:
                id2score[idx].extend(reward_list)
            else:
                avg_score = torch.stack(reward_list).mean()
                traj2avg[(idx, t_idx)] = avg_score
                id2score[idx].append(avg_score)
        if not group_by_agent_id:
            for i in range(bsz):
                scores[i] = traj2avg[(index[i], traj_index[i])]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_opd_distill_signal(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    opd_lambda: float = 1.0,
    power_opd_enabled: bool = False,
    power_alpha: float = 1.0,
):
    """Compute the per-token OPD distillation signal (the OPD token-level "reward").

    Vanilla OPD uses the unbounded log-ratio reward
        A_t = log π_T(a_t) - log π_θ(a_t)
    which PowerOPD (arxiv 2606.17199) shows produces high-variance gradients
    concentrated at early positions. PowerOPD replaces it with the Box-Cox
    power difference, a natively bounded, sign-consistent reward:
        A_t = π_T^α - π_θ^α = exp(α·log π_T) - exp(α·log π_θ),   α > 0
    which lies in [-1, 1]. The 1/α factor of the Box-Cox transform is absorbed
    into the learning rate / λ. As α → 0 the power difference recovers the
    vanilla log-ratio.

    Args:
        teacher_log_probs: (bs, resp_len) log π_T (privileged-context teacher).
        student_log_probs: (bs, resp_len) log π_θ (student/old policy).
        response_mask: (bs, resp_len) mask (1 → keep, 0 → zero the signal).
        opd_lambda: scaling factor (ExOPD λ), applied after the transform.
        power_opd_enabled: use the PowerOPD power difference instead of log-ratio.
        power_alpha: Box-Cox exponent α > 0 (only used when power_opd_enabled).

    Returns:
        (bs, resp_len) stop-gradient per-token distillation signal.
    """
    with torch.no_grad():
        if power_opd_enabled:
            # exp(α·log π) ∈ (0, 1] for α > 0 and log π ≤ 0, so the difference
            # is bounded in [-1, 1] with no overflow.
            signal = torch.exp(power_alpha * teacher_log_probs) - torch.exp(power_alpha * student_log_probs)
        else:
            signal = teacher_log_probs - student_log_probs
        signal = signal * response_mask
        return opd_lambda * signal


def compute_opd_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    opd_lambda: float = 1.0,
    combine_with_grpo: bool = False,
    grpo_advantages: torch.Tensor = None,
    grpo_weight: float = 0.0,
    power_opd_enabled: bool = False,
    power_alpha: float = 1.0,
):
    """
    Compute per-token OPD advantage via on-policy context distillation.

    A_t = teacher_log_prob(a_t|h_teacher) - student_log_prob(a_t|h_self)

    When teacher (with privileged future context) assigns higher probability to a token
    than the student, A_t > 0, encouraging the student to increase that token's probability.

    Args:
        student_log_probs: (bs, response_length) — log π_θ(a_t | h_self, a_{<t})
        teacher_log_probs: (bs, response_length) — log π_θ(a_t | h_teacher, a_{<t}), stop-grad
        response_mask: (bs, response_length)
        opd_lambda: scaling factor (ExOPD: lambda > 1 can push beyond teacher upper bound)
        combine_with_grpo: whether to linearly combine with GRPO advantages
        grpo_advantages: pre-computed GRPO advantages (bs, response_length)
        grpo_weight: weight for GRPO component when combining

    Returns:
        advantages: (bs, response_length) — per-token dense advantage
        returns: (bs, response_length) — same as advantages
    """
    with torch.no_grad():
        # teacher - student: positive when teacher assigns higher prob → encourage token
        opd_adv = compute_opd_distill_signal(
            teacher_log_probs,
            student_log_probs,
            response_mask,
            opd_lambda,
            power_opd_enabled=power_opd_enabled,
            power_alpha=power_alpha,
        )

        if combine_with_grpo and grpo_advantages is not None:
            advantages = (1.0 - grpo_weight) * opd_adv + grpo_weight * grpo_advantages
        else:
            advantages = opd_adv

    return advantages, advantages


def compute_opd_grpo_joint_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    grpo_response_mask: torch.Tensor = None,
    opd_beta: float = 5e-3,
    opd_lambda: float = 1.0,
    grpo_outcome_weight: float = 1.0,
    index: np.ndarray = None,
    traj_index: np.ndarray = None,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
    power_opd_enabled: bool = False,
    power_alpha: float = 1.0,
):
    """Compute the additive OPD-GRPO joint optimization target (ADV-level).

    Serves the ``opd_grpo_joint_adv`` advantage estimator: OPD and GRPO are
    combined at the ADVANTAGE level into a single ``advantages`` tensor, which
    the actor consumes via one ``compute_policy_loss`` call. Consequently OPD
    and GRPO share the same PPO importance ratio / clip boundary — when the
    GRPO ratio is clipped, the OPD distillation gradient is clipped too.

    Formulation (mirrors the reference OPD project)::

        A_joint = beta * lambda * A_OPD + w_grpo * A_GRPO

    ``opd_beta`` is scheduled by the trainer (``opd.beta_*``). Keeping the two
    coefficients independent lets the OPD regularizer decay without shrinking
    the outcome optimization signal.

    For the LOSS-level variant (two separate loss terms, OPD via pure policy
    gradient without ratio/clip), see
    :func:`compute_opd_grpo_joint_loss_advantages`.
    """
    with torch.no_grad():
        opd_adv = compute_opd_distill_signal(
            teacher_log_probs,
            student_log_probs,
            response_mask,
            opd_lambda,
            power_opd_enabled=power_opd_enabled,
            power_alpha=power_alpha,
        )

        if grpo_response_mask is None:
            grpo_response_mask = response_mask
        grpo_adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=grpo_response_mask,
            index=index,
            traj_index=traj_index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )

        advantages = opd_beta * opd_adv + grpo_outcome_weight * grpo_adv

    return advantages, advantages


def compute_opd_grpo_joint_loss_advantages(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    grpo_response_mask: torch.Tensor = None,
    opd_lambda: float = 1.0,
    index: np.ndarray = None,
    traj_index: np.ndarray = None,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
    power_opd_enabled: bool = False,
    power_alpha: float = 1.0,
):
    """Compute the LOSS-level OPD-GRPO joint target: return the two advantages SEPARATELY.

    Serves the ``opd_grpo_joint_loss`` advantage estimator. Unlike
    :func:`compute_opd_grpo_joint_advantage` which fuses OPD+GRPO into one
    ``advantages`` tensor (advantage-level), this returns a 2-tuple
    ``(adv_opd, adv_grpo)`` and leaves the combination to the actor, where the
    two become TWO separate loss terms:

      * GRPO term: standard PPO dual-clip surrogate via ``compute_policy_loss``
        with ``adv_grpo`` (uses importance ratio + clipping).
      * OPD term: pure policy gradient ``agg_loss(-adv_opd * log_prob)`` — NO
        ratio, NO clipping. The distillation gradient therefore flows
        independently of whether the GRPO ratio is clipped.

    Both tensors are ``stop-gradient`` (computed under ``torch.no_grad()``);
    only ``log_prob`` (the actor's current policy) carries gradient in the loss.

    Args:
        student_log_probs: (bs, resp_len) — ``old_log_probs`` (student/old policy).
        teacher_log_probs: (bs, resp_len) — privileged-context teacher (no_grad).
        token_level_rewards: (bs, resp_len) — outcome rewards for GRPO.
        response_mask: (bs, resp_len) — used for the OPD per-token term.
        grpo_response_mask: optional separate mask for GRPO normalization
            (multi_turn uses ``loss_mask[:, -resp_len:]``); defaults to
            ``response_mask``.
        opd_lambda: scaling for the OPD term (``opd.lambda``, shared with adv level).
        index: GRPO group key (``uid`` or ``f"{uid}_{agent_id}"``).
        traj_index: per-trajectory id (``traj_uid``).
        norm_adv_by_std_in_grpo, group_by_agent_id: forwarded to GRPO.

    Returns:
        (adv_opd, adv_grpo): each (bs, resp_len), stop-gradient. The driver
        stores them as ``batch["advantages_opd"]`` / ``batch["advantages_grpo"]``
        and lets the actor combine the two loss terms.
    """
    with torch.no_grad():
        adv_opd = compute_opd_distill_signal(
            teacher_log_probs,
            student_log_probs,
            response_mask,
            opd_lambda,
            power_opd_enabled=power_opd_enabled,
            power_alpha=power_alpha,
        )

        if grpo_response_mask is None:
            grpo_response_mask = response_mask
        adv_grpo, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=grpo_response_mask,
            index=index,
            traj_index=traj_index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )

    return adv_opd, adv_grpo


def compute_opd_last_outcome_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    is_last_step: torch.Tensor,
    opd_lambda: float = 1.0,
    index: np.ndarray = None,
    traj_index: np.ndarray = None,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
    power_opd_enabled: bool = False,
    power_alpha: float = 1.0,
):
    """
    Mode 1: Last step uses outcome reward advantage, all other steps use OPD.

    For non-last steps: A_t = lambda * (teacher_lp - student_lp)
    For last step: A_t = GRPO outcome advantage (normalized within group)
    """
    with torch.no_grad():
        opd_adv = compute_opd_distill_signal(
            teacher_log_probs,
            student_log_probs,
            response_mask,
            opd_lambda,
            power_opd_enabled=power_opd_enabled,
            power_alpha=power_alpha,
        )

        grpo_adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )

        is_last = is_last_step.unsqueeze(-1).expand_as(opd_adv)
        advantages = torch.where(is_last, grpo_adv, opd_adv)

    return advantages, advantages


def compute_opd_hybrid_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    opd_lambda: float = 1.0,
    outcome_weight: float = 0.3,
    index: np.ndarray = None,
    traj_index: np.ndarray = None,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
    power_opd_enabled: bool = False,
    power_alpha: float = 1.0,
):
    """
    Mode 2: All steps use a mix of OPD dense reward + outcome reward.

    A_t = (1 - outcome_weight) * OPD_adv + outcome_weight * GRPO_adv
    """
    with torch.no_grad():
        opd_adv = compute_opd_distill_signal(
            teacher_log_probs,
            student_log_probs,
            response_mask,
            opd_lambda,
            power_opd_enabled=power_opd_enabled,
            power_alpha=power_alpha,
        )

        grpo_adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )

        advantages = (1.0 - outcome_weight) * opd_adv + outcome_weight * grpo_adv

    return advantages, advantages


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, group_by_agent_id: bool = False):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    group_by_agent_id: bool = False,
    return_metrics: bool = False,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    del epsilon, group_by_agent_id  # grouping is fully described by ``index`` and ``traj_index``
    response_scores = token_level_rewards.sum(dim=-1)
    advantages_per_response = torch.zeros_like(response_scores)
    trajectory_scores = defaultdict(lambda: torch.zeros((), device=response_scores.device, dtype=response_scores.dtype))
    with torch.no_grad():
        bsz = response_scores.shape[0]
        for i in range(bsz):
            # Multiple responses by the same role in one trajectory form one LOO sample.
            trajectory_scores[(index[i], traj_index[i])] += response_scores[i]

        group_trajectories = defaultdict(list)
        for (group_id, trajectory_id), score in trajectory_scores.items():
            group_trajectories[group_id].append((trajectory_id, score))

        baselines = {}
        trajectory_advantages = {}
        group_sizes = []
        for group_id, entries in group_trajectories.items():
            group_size = len(entries)
            group_sizes.append(group_size)
            total = torch.stack([score for _, score in entries]).sum()
            for trajectory_id, score in entries:
                baseline = (total - score) / (group_size - 1) if group_size > 1 else torch.zeros_like(score)
                baselines[(group_id, trajectory_id)] = baseline
                trajectory_advantages[(group_id, trajectory_id)] = score - baseline if group_size > 1 else torch.zeros_like(score)

        for i in range(bsz):
            advantages_per_response[i] = trajectory_advantages[(index[i], traj_index[i])]
        advantages = advantages_per_response.unsqueeze(-1) * response_mask

        trajectory_reward_values = torch.stack(list(trajectory_scores.values())) if trajectory_scores else response_scores.new_zeros(1)
        baseline_values = torch.stack(list(baselines.values())) if baselines else response_scores.new_zeros(1)
        trajectory_advantage_values = (
            torch.stack(list(trajectory_advantages.values())) if trajectory_advantages else response_scores.new_zeros(1)
        )
        metrics = {
            "rloo/group_size_mean": float(np.mean(group_sizes)) if group_sizes else 0.0,
            "rloo/singleton_group_fraction": float(np.mean(np.asarray(group_sizes) < 2)) if group_sizes else 0.0,
            "rloo/reward_mean": trajectory_reward_values.mean().item(),
            "rloo/baseline_mean": baseline_values.mean().item(),
            "rloo/advantage_mean": trajectory_advantage_values.mean().item(),
            "rloo/advantage_std": trajectory_advantage_values.std(unbiased=False).item(),
            "rloo/nonzero_adv_fraction": (trajectory_advantage_values.abs() > 0).float().mean().item(),
        }

    if return_metrics:
        return advantages, advantages, metrics
    return advantages, advantages


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = torch.zeros(
            token_level_rewards.shape[0], device=token_level_rewards.device, dtype=token_level_rewards.dtype
        )

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        valid_count = response_mask.sum()
        if valid_count > 1:
            advantages = verl_F.masked_whiten(returns, response_mask)
        else:
            # masked_whiten intentionally rejects a one-token sample; an RL batch must
            # remain finite even for an empty or single-token response.
            advantages = torch.zeros_like(returns)
        advantages = advantages * response_mask

    return advantages, returns


def compute_gspo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
):
    """GSPO objective with one length-normalized importance ratio per response."""
    token_count = response_mask.sum(dim=-1)
    valid_sequence = token_count > 0
    old_log_prob = old_log_prob.detach()
    sequence_log_ratio = ((log_prob - old_log_prob) * response_mask).sum(dim=-1) / token_count.clamp_min(1)
    sequence_ratio = torch.exp(sequence_log_ratio)
    sequence_advantage = (advantages * response_mask).sum(dim=-1) / token_count.clamp_min(1)
    clipped_ratio = torch.clamp(sequence_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    losses = -torch.minimum(sequence_ratio * sequence_advantage, clipped_ratio * sequence_advantage)
    if valid_sequence.any():
        pg_loss = losses[valid_sequence].mean()
        ratios = sequence_ratio[valid_sequence]
        log_ratios = sequence_log_ratio[valid_sequence]
        adv = sequence_advantage[valid_sequence]
        clipped_low = (ratios < 1.0 - clip_ratio_low) & (adv < 0)
        clipped_high = (ratios > 1.0 + clip_ratio_high) & (adv > 0)
        metrics = {
            "gspo/sequence_ratio_mean": ratios.mean().detach().item(),
            "gspo/sequence_ratio_std": ratios.std(unbiased=False).detach().item(),
            "gspo/sequence_ratio_min": ratios.min().detach().item(),
            "gspo/sequence_ratio_max": ratios.max().detach().item(),
            "gspo/sequence_clipfrac": (clipped_low | clipped_high).float().mean().detach().item(),
            "gspo/sequence_clipfrac_low": clipped_low.float().mean().detach().item(),
            "gspo/sequence_clipfrac_high": clipped_high.float().mean().detach().item(),
            "gspo/approx_kl": (-log_ratios).mean().detach().item(),
            "gspo/pg_loss": pg_loss.detach().item(),
        }
    else:
        pg_loss = log_prob.sum() * 0.0
        metrics = {key: 0.0 for key in (
            "gspo/sequence_ratio_mean", "gspo/sequence_ratio_std", "gspo/sequence_ratio_min",
            "gspo/sequence_ratio_max", "gspo/sequence_clipfrac", "gspo/sequence_clipfrac_low",
            "gspo/sequence_clipfrac_high", "gspo/approx_kl", "gspo/pg_loss",
        )}
    return pg_loss, metrics


def compute_dapo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
):
    """DAPO Clip-Higher objective with token-level (token-mean) aggregation."""
    log_ratio = log_prob - old_log_prob.detach()
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    loss_mat = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
    pg_loss = agg_loss(loss_mat, response_mask, "token-mean")
    low = (ratio < 1.0 - clip_ratio_low) & (advantages < 0) & response_mask.bool()
    high = (ratio > 1.0 + clip_ratio_high) & (advantages > 0) & response_mask.bool()
    denom = response_mask.sum().clamp_min(1)
    metrics = {
        "dapo/clipfrac_low": (low.float().sum() / denom).detach().item(),
        "dapo/clipfrac_high": (high.float().sum() / denom).detach().item(),
        "dapo/pg_loss": pg_loss.detach().item(),
    }
    return pg_loss, metrics


def apply_dapo_overlong_reward_shaping(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    max_response_length: int,
    cache_len: int,
    response_lengths: Optional[torch.Tensor] = None,
):
    """Add DAPO's soft overlong penalty to the terminal valid response token."""
    if cache_len < 0 or cache_len >= max_response_length:
        raise ValueError("DAPO overlong_cache_len must be 0 or satisfy 0 < cache_len < max_response_length")
    lengths = response_mask.sum(dim=-1) if response_lengths is None else response_lengths.to(response_mask.device)
    correctness = token_level_rewards.sum(dim=-1)
    penalties = torch.zeros_like(correctness)
    if cache_len > 0:
        threshold = max_response_length - cache_len
        penalties = torch.where(
            lengths <= threshold,
            penalties,
            torch.where(
                lengths <= max_response_length,
                (threshold - lengths).to(correctness.dtype) / cache_len,
                -torch.ones_like(correctness),
            ),
        )
    shaped = token_level_rewards.clone()
    valid = response_mask.bool().any(dim=-1)
    if valid.any():
        last_indices = response_mask.sum(dim=-1).long().clamp_min(1) - 1
        rows = torch.arange(shaped.shape[0], device=shaped.device)[valid]
        shaped[rows, last_indices[valid]] += penalties[valid]
    metrics = {
        "dapo/correctness_reward_mean": correctness.mean().item() if correctness.numel() else 0.0,
        "dapo/overlong_penalty_mean": penalties.mean().item() if penalties.numel() else 0.0,
        "dapo/overlong_penalty_fraction": (penalties != 0).float().mean().item() if penalties.numel() else 0.0,
        "dapo/final_reward_mean": (correctness + penalties).mean().item() if correctness.numel() else 0.0,
    }
    return shaped, penalties, metrics


def select_dapo_mixed_query_groups(
    query_uids: np.ndarray,
    traj_uids: np.ndarray,
    trajectory_rewards: np.ndarray,
):
    """Return all response rows belonging to query groups with both success and failure trajectories."""
    query_to_trajectories = defaultdict(dict)
    for query_uid, traj_uid, reward in zip(query_uids, traj_uids, trajectory_rewards):
        # A trajectory may contain multiple agent responses. Its terminal rule
        # reward can appear on only one row, so max avoids duplicate counting.
        old = query_to_trajectories[query_uid].get(traj_uid, -np.inf)
        query_to_trajectories[query_uid][traj_uid] = max(old, float(reward))
    accepted = set()
    rejected_all_zero = rejected_all_one = 0
    for query_uid, rewards_by_traj in query_to_trajectories.items():
        successes = np.asarray(list(rewards_by_traj.values())) > 0
        if successes.any() and not successes.all():
            accepted.add(query_uid)
        elif successes.all():
            rejected_all_one += 1
        else:
            rejected_all_zero += 1
    indices = np.flatnonzero(np.isin(query_uids, list(accepted)))
    stats = {
        "accepted_prompt_groups": len(accepted),
        "rejected_all_zero": rejected_all_zero,
        "rejected_all_one": rejected_all_one,
        "generated_prompt_groups": len(query_to_trajectories),
    }
    return indices, stats


def validate_dapo_dynamic_sampling_result(
    collected_trajectories: int,
    world_size: int,
    generation_rounds: int,
    max_generation_rounds: int,
):
    """Validate the usable partial batch when DAPO reaches its generation cap."""
    if collected_trajectories == 0:
        raise ValueError(
            f"DAPO dynamic sampling found no mixed-reward prompt groups after {max_generation_rounds} generation rounds."
        )
    if collected_trajectories % world_size != 0:
        raise ValueError(
            "DAPO dynamic sampling produced an incomplete distributed batch: "
            f"{collected_trajectories} trajectories after {generation_rounds} rounds is not divisible "
            f"by world_size={world_size}."
        )


def dapo_dynamic_sampling_enabled(adv_estimator, dynamic_sampling: bool, is_train: bool) -> bool:
    """DAPO dynamic sampling is a training-only operation, never a validation operation."""
    estimator_name = str(getattr(adv_estimator, "value", adv_estimator))
    return bool(is_train and dynamic_sampling and estimator_name == "dapo")


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def resolve_joint_loss_agg_modes(actor_config):
    """Resolve joint-loss aggregation modes with backward-compatible fallbacks."""
    loss_agg_mode = actor_config.get("loss_agg_mode", "token-mean")
    return (
        actor_config.get("grpo_loss_agg_mode", loss_agg_mode),
        actor_config.get("opd_loss_agg_mode", loss_agg_mode),
    )


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        token_counts = torch.sum(loss_mask, dim=-1)
        valid_sequences = token_counts > 0
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / token_counts.clamp_min(1)
        loss = torch.sum(seq_losses[valid_sequences]) / valid_sequences.sum().clamp_min(1)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_fact_loo_metrics(
    full_log_probs: torch.Tensor,
    without_log_probs: torch.Tensor,
    advantages_grpo: torch.Tensor,
    loss_mask: torch.Tensor,
    conflict_weight: float = 1.0,
    eps: float = 1e-8,
):
    """Compute target-level fact LOO statistics on the actor loss mask."""
    delta = (full_log_probs - without_log_probs) * loss_mask
    signed = advantages_grpo * delta
    denominator = loss_mask.sum().clamp_min(1)
    aligned = (torch.relu(signed) * loss_mask).sum() / denominator
    conflict = (torch.relu(-signed) * loss_mask).sum() / denominator
    effect_rms = torch.sqrt((delta.square() * loss_mask).sum() / denominator)
    net = aligned - float(conflict_weight) * conflict
    precision = aligned / (aligned + conflict + eps)
    return {
        "aligned_mass": aligned,
        "conflict_mass": conflict,
        "net_alignment": net,
        "directional_precision": precision,
        "effect_rms": effect_rms,
    }


def rlsd_alpha(global_step: int, alpha_init: float, alpha_final: float, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return float(alpha_final)
    progress = min(max(float(global_step), 0.0) / float(anneal_steps), 1.0)
    return float(alpha_init) + progress * (float(alpha_final) - float(alpha_init))


def compute_rlsd_advantage(
    advantages_grpo: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    old_student_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    has_teacher: torch.Tensor,
    eps_w: float,
    alpha: float,
):
    """RLSD multiplicative credit that preserves the GRPO advantage sign."""
    if not 0.0 < eps_w < 1.0:
        raise ValueError(f"rlsd.eps_w must be in (0, 1), got {eps_w}")
    teacher_mask = has_teacher.to(loss_mask.dtype)
    if teacher_mask.ndim == 1:
        teacher_mask = teacher_mask.unsqueeze(-1)
    active_mask = loss_mask * teacher_mask
    log_credit = torch.sign(advantages_grpo) * (teacher_log_probs - old_student_log_probs)
    lower = math.log(1.0 - eps_w)
    upper = math.log(1.0 + eps_w)
    clipped = torch.clamp(log_credit, min=lower, max=upper)
    credit = torch.where(active_mask.bool(), torch.exp(clipped), torch.ones_like(clipped))
    multiplier = (1.0 - float(alpha)) + float(alpha) * credit
    advantages_rlsd = advantages_grpo * multiplier * loss_mask
    return advantages_rlsd, credit, multiplier, log_credit


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
