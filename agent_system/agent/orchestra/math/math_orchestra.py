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

from __future__ import annotations
"""Agent execution structures – chain & hierarchy.
"""
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.orchestra.base import BaseOrchestra
# from agent_system.agent.agents import *
import importlib
from verl import DataProto
import numpy as np


def update_team_context(agent_id: str, team_context: List[str], text_response: str, agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the observation dictionary with the text response."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(team_context), dtype=bool)
    # Naive append of the latest responses to observations
    for i in range(len(team_context)):
        if agent_active_mask[i]:
            team_context[i] = team_context[i] + f"""\nThe output of "{agent_id}": {text_response[i]}\n"""
    return team_context

def update_text_action(text_actions: List[str], text_response: List[str], agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the text actions with the latest response."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(text_actions), dtype=bool)

    for i in range(len(text_actions)):
        if agent_active_mask[i]:
            text_actions[i] = text_response[i]
    return text_actions

class MathMultiAgentOrchestra(BaseOrchestra):
    SOLVER_AGENT = "Solver Agent"
    VERIFIER_AGENT = "Verifier Agent"
    JUDGER_AGENT = "Judger Agent"

    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        importlib.import_module("agent_system.agent.agents.math")
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        if not self.agents:
            raise ValueError("SolverVerifierMathOrchestra requires at least one agent.")

        self.agent_order = self.agent_ids
        self.max_loop_num = getattr(self.config.agent.orchestra.math, "max_loop_num", 3)

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs, active_masks: np.ndarray, step: int, ground_truths=None, questions=None) -> Tuple[List[str], Dict[str, DataProto]]:
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)

        approved_vector = np.zeros(len(gen_batch), dtype=bool)

        for loop_i in range(self.max_loop_num):
            # Solver runs on not-yet-approved items
            solver_mask = np.logical_and(active_masks, np.logical_not(approved_vector)).astype(bool)
            if solver_mask.any() and self.SOLVER_AGENT in self.agents:
                actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.SOLVER_AGENT]]
                batch, text_repsonses = self.agents[self.SOLVER_AGENT].call(
                    gen_batch=gen_batch,
                    env_obs=env_obs,
                    team_context=team_context,
                    actor_rollout_wg=actor_rollout_wg,
                    agent_active_mask=solver_mask,
                    step=step,
                )
                team_context = update_team_context(self.SOLVER_AGENT, team_context, text_repsonses, solver_mask)
                self.save_to_buffer(self.SOLVER_AGENT, batch)
                text_actions = update_text_action(text_actions, text_repsonses, solver_mask)

            if loop_i == self.max_loop_num - 1:
                break
            # Verifier checks those items (skip the last loop)
            verifier_mask = np.logical_and(active_masks, np.logical_not(approved_vector)).astype(bool)
            if verifier_mask.any() and self.VERIFIER_AGENT in self.agents:
                actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.VERIFIER_AGENT]]
                batch, text_repsonses = self.agents[self.VERIFIER_AGENT].call(
                    gen_batch=gen_batch,
                    env_obs=env_obs,
                    team_context=team_context,
                    actor_rollout_wg=actor_rollout_wg,
                    agent_active_mask=verifier_mask,
                    step=step,
                )
                team_context = update_team_context(self.VERIFIER_AGENT, team_context, text_repsonses, verifier_mask)
                self.save_to_buffer(self.VERIFIER_AGENT, batch)

                # Update approvals
                approved_vector = self.agents[self.VERIFIER_AGENT].update_approved_vector(text_repsonses, approved_vector, verifier_mask)

            if approved_vector.all():
                break

        # === Judger Phase (改进1): GT-based trajectory-level evaluation ===
        # Note: hydra may store the flag as a str "False" (truthy) when it comes
        # from a shell env var, so normalize to a real bool before checking.
        _judger_flag = str(getattr(self.config.algorithm.opd, 'use_judger', False)).strip().lower()
        use_judger = (
            getattr(self.config.algorithm, 'opd', None) is not None
            and _judger_flag in ("true", "1")
            and self.JUDGER_AGENT in self.agents
            and ground_truths is not None
        )
        if use_judger:
            judger = self.agents[self.JUDGER_AGENT]
            batch_size = len(gen_batch)

            # Build trajectory texts for each active item
            trajectories = [""] * batch_size
            problems = [""] * batch_size
            gt_list = [""] * batch_size
            for i in range(batch_size):
                if active_masks[i]:
                    problem_text = env_obs['text'][i] if env_obs and 'text' in env_obs else ""
                    trajectories[i] = team_context[i]
                    problems[i] = questions[i] if questions else problem_text
                    gt_list[i] = ground_truths[i] if ground_truths else ""

            # Resume GPU filler — Judger API calls are CPU-bound, GPU is idle
            try:
                from scripts.gpu_filler_control import resume as gpu_filler_resume
                gpu_filler_resume()
            except Exception:
                pass

            # Parallel API calls via ThreadPoolExecutor
            judgments = judger.judge_batch(trajectories, problems, gt_list)

            # Append judgment J to team_context
            for i in range(batch_size):
                if active_masks[i] and judgments[i]:
                    team_context[i] += (
                        f'\n[Judger Assessment]\n{judgments[i]}\n'
                    )

            # Store Judger output as a synthetic step in the buffer
            # so OPD teacher construction can find and inject J
            judger_batch = self._build_judger_batch(judgments, gen_batch)
            self.save_to_buffer(self.JUDGER_AGENT, judger_batch)

        return text_actions, self.multiagent_batch_buffer

    def _build_judger_batch(self, judgments: List[str], gen_batch: DataProto) -> DataProto:
        """Build a synthetic DataProto for Judger outputs.

        The Judger does not produce real LLM responses/generate gradient — this
        creates a minimal DataProto that carries the judgment text so it can be
        stored in multiagent_batch_buffer and later found by the OPD teacher via
        total_batch_list (for J injection into other steps' teacher context).

        Design:
          - agent_active_mask=True: enters total_batch_list (so OPD teacher can
            find it by agent_id == "Judger Agent" and extract J).
          - active_masks=False: does NOT enter effective_batch, so it never
            participates in training (no gradient, no forward pass on fake tokens).
          - All tensor fields have a dummy 1-token shape so to_list_of_dict()
            (which indexes by 'input_ids') works without KeyError.
        """
        import torch
        batch_size = len(gen_batch)
        dummy = torch.zeros(batch_size, 1, dtype=torch.long)
        batch = DataProto.from_dict(
            tensors={
                "input_ids": dummy.clone(),
                "attention_mask": dummy.clone(),
                "position_ids": dummy.clone(),
                "responses": dummy.clone(),
                "prompts": dummy.clone(),
            },
            non_tensors={
                "text_response": np.array(judgments, dtype=object),
                "agent_active_mask": np.ones(batch_size, dtype=bool),
                "active_masks": np.zeros(batch_size, dtype=bool),
            },
        )
        return batch