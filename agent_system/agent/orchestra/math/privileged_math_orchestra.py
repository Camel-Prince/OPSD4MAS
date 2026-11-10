"""
Privileged Math Orchestra — MAS rollout with oracle privilege injection.

Extends MathMultiAgentOrchestra to inject oracle privilege info (facts, Judger
assessment) into agent prompts during the Solver-Verifier loop. Used for OPD
ablation experiments comparing MAS performance under different privilege levels.

The privilege info format matches construct_teacher_data in opd_teacher.py:
  [Privileged Facts]
  - fact 1
  - fact 2

  [Judger Assessment]
  judgment text

Step-by-step fact filtering reuses FactFilter.filter_for_step() from opd_facts.py.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from agent_system.agent.orchestra.math.math_orchestra import (
    MathMultiAgentOrchestra,
    update_team_context,
    update_text_action,
)
from verl import DataProto


class PrivilegeSetting(Enum):
    """Which types of oracle privilege info to inject during rollout."""
    NONE = "none"           # No privilege — plain MAS rollout (baseline)
    FULL = "full"           # Judger + Facts
    JUDGER_ONLY = "judger"  # Judger assessment only
    FACTS_ONLY = "facts"    # Step-filtered facts only


@dataclass
class OraclePrivilegeData:
    """Oracle privilege info from a Phase 1 reference trajectory."""
    fact_pool: list = field(default_factory=list)  # List[Fact]
    judger_text: str = ""


class PrivilegedMathOrchestra(MathMultiAgentOrchestra):
    """Math orchestra that injects oracle privilege info into agent prompts.

    Before each Solver/Verifier call, builds a privileged copy of team_context
    with filtered facts and/or Judger assessment prepended. The original
    team_context is updated with agent outputs only (privilege does not persist).
    """

    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: dict,
        tokenizers: dict = None,
        processors: dict = None,
        config=None,
        privilege_data_list: Optional[List[Optional[OraclePrivilegeData]]] = None,
        fact_filter=None,
        setting: PrivilegeSetting = PrivilegeSetting.NONE,
    ):
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        self.privilege_data_list = privilege_data_list or []
        self.fact_filter = fact_filter
        self.setting = setting

    # ------------------------------------------------------------------
    # Run (override)
    # ------------------------------------------------------------------

    def run(
        self,
        gen_batch: DataProto,
        env_obs: dict,
        actor_rollout_wgs,
        active_masks: np.ndarray,
        step: int,
        ground_truths=None,
        questions=None,
    ):
        """Run Solver-Verifier-Judger loop with optional privilege injection.

        For NONE setting, delegates entirely to parent (no overhead).
        For other settings, injects oracle privilege into team_context copies
        before each agent call, using step-by-step FactFilter.filter_for_step().
        """
        if self.setting == PrivilegeSetting.NONE:
            return super().run(
                gen_batch=gen_batch,
                env_obs=env_obs,
                actor_rollout_wgs=actor_rollout_wgs,
                active_masks=active_masks,
                step=step,
                ground_truths=ground_truths,
                questions=questions,
            )

        # ── Privileged path ──────────────────────────────────────
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)

        approved_vector = np.zeros(len(gen_batch), dtype=bool)
        batch_size = len(gen_batch)

        # Track agent step index per sample for fact filtering.
        # Phase 1 facts use step_idx matching trajectory position (0=S₀, 1=V₁, 2=S₂, …).
        agent_step_counter = np.zeros(batch_size, dtype=int)

        for loop_i in range(self.max_loop_num):
            # ── Solver ──────────────────────────────────────────
            solver_mask = np.logical_and(
                active_masks, np.logical_not(approved_vector)
            ).astype(bool)

            if solver_mask.any() and self.SOLVER_AGENT in self.agents:
                # Build privileged team_context copy
                priv_team_context = self._build_privileged_team_context(
                    team_context=team_context,
                    active_mask=solver_mask,
                    current_step_indices=agent_step_counter,
                    agent_role=self.SOLVER_AGENT,
                )

                actor_rollout_wg = actor_rollout_wgs[
                    self.agents_to_wg_mapping[self.SOLVER_AGENT]
                ]
                batch, text_responses = self.agents[self.SOLVER_AGENT].call(
                    gen_batch=gen_batch,
                    env_obs=env_obs,
                    team_context=priv_team_context,
                    actor_rollout_wg=actor_rollout_wg,
                    agent_active_mask=solver_mask,
                    step=step,
                )
                # Update ORIGINAL team_context (privilege does NOT persist)
                team_context = update_team_context(
                    self.SOLVER_AGENT, team_context, text_responses, solver_mask
                )
                self.save_to_buffer(self.SOLVER_AGENT, batch)
                text_actions = update_text_action(
                    text_actions, text_responses, solver_mask
                )
                # Advance step counter for active solver samples
                agent_step_counter[solver_mask] += 1

            if loop_i == self.max_loop_num - 1:
                break

            # ── Verifier ─────────────────────────────────────────
            verifier_mask = np.logical_and(
                active_masks, np.logical_not(approved_vector)
            ).astype(bool)

            if verifier_mask.any() and self.VERIFIER_AGENT in self.agents:
                priv_team_context = self._build_privileged_team_context(
                    team_context=team_context,
                    active_mask=verifier_mask,
                    current_step_indices=agent_step_counter,
                    agent_role=self.VERIFIER_AGENT,
                )

                actor_rollout_wg = actor_rollout_wgs[
                    self.agents_to_wg_mapping[self.VERIFIER_AGENT]
                ]
                batch, text_responses = self.agents[self.VERIFIER_AGENT].call(
                    gen_batch=gen_batch,
                    env_obs=env_obs,
                    team_context=priv_team_context,
                    actor_rollout_wg=actor_rollout_wg,
                    agent_active_mask=verifier_mask,
                    step=step,
                )
                team_context = update_team_context(
                    self.VERIFIER_AGENT, team_context, text_responses, verifier_mask
                )
                self.save_to_buffer(self.VERIFIER_AGENT, batch)

                approved_vector = self.agents[self.VERIFIER_AGENT].update_approved_vector(
                    text_responses, approved_vector, verifier_mask
                )
                agent_step_counter[verifier_mask] += 1

            if approved_vector.all():
                break

        # ── Judger Phase (unchanged from parent) ─────────────────
        _judger_flag = str(getattr(self.config.algorithm.opd, 'use_judger', False)).strip().lower()
        use_judger = (
            getattr(self.config.algorithm, 'opd', None) is not None
            and _judger_flag in ("true", "1")
            and self.JUDGER_AGENT in self.agents
            and ground_truths is not None
        )
        if use_judger:
            judger = self.agents[self.JUDGER_AGENT]

            trajectories = [""] * batch_size
            problems = [""] * batch_size
            gt_list = [""] * batch_size
            for i in range(batch_size):
                if active_masks[i]:
                    problem_text = env_obs['text'][i] if env_obs and 'text' in env_obs else ""
                    trajectories[i] = team_context[i]
                    problems[i] = questions[i] if questions else problem_text
                    gt_list[i] = ground_truths[i] if ground_truths else ""

            try:
                from scripts.gpu_filler_control import resume as gpu_filler_resume
                gpu_filler_resume()
            except Exception:
                pass

            judgments = judger.judge_batch(trajectories, problems, gt_list)

            for i in range(batch_size):
                if active_masks[i] and judgments[i]:
                    team_context[i] += (
                        f'\n[Judger Assessment]\n{judgments[i]}\n'
                    )

            judger_batch = self._build_judger_batch(judgments, gen_batch)
            self.save_to_buffer(self.JUDGER_AGENT, judger_batch)

        return text_actions, self.multiagent_batch_buffer

    # ------------------------------------------------------------------
    # Privilege helpers
    # ------------------------------------------------------------------

    def _build_privileged_team_context(
        self,
        team_context: List[str],
        active_mask: np.ndarray,
        current_step_indices: np.ndarray,
        agent_role: str,
    ) -> List[str]:
        """Build a copy of team_context with oracle privilege text prepended.

        Only modifies entries where active_mask is True. The original
        team_context is never mutated.
        """
        priv_context = copy.deepcopy(team_context)
        for i in range(len(priv_context)):
            if active_mask[i]:
                privilege_text = self._get_privilege_text(
                    env_idx=i,
                    current_step=int(current_step_indices[i]),
                    agent_role=agent_role,
                )
                if privilege_text:
                    priv_context[i] = privilege_text + "\n\n" + priv_context[i]
        return priv_context

    def _get_privilege_text(
        self, env_idx: int, current_step: int, agent_role: str
    ) -> str:
        """Build privilege text using FactFilter + Judger, matching
        construct_teacher_data format from opd_teacher.py.

        Returns empty string for NONE setting or when no oracle data exists.
        """
        if self.setting == PrivilegeSetting.NONE:
            return ""

        if env_idx >= len(self.privilege_data_list):
            return ""
        pd = self.privilege_data_list[env_idx]
        if pd is None:
            return ""

        parts = []

        # Facts: step-filtered via FactFilter (matches construct_teacher_data)
        if self.setting in (PrivilegeSetting.FULL, PrivilegeSetting.FACTS_ONLY):
            if pd.fact_pool and self.fact_filter is not None:
                try:
                    relevant = self.fact_filter.filter_for_step(
                        fact_pool=pd.fact_pool,
                        current_step_idx=current_step,
                        agent_role=agent_role,
                    )
                    if relevant:
                        parts.append(
                            "[Privileged Facts]\n"
                            + "\n".join(f"- {t}" for t in relevant)
                        )
                except Exception:
                    pass

        # Judger: full assessment text (matches construct_teacher_data Judger path)
        if self.setting in (PrivilegeSetting.FULL, PrivilegeSetting.JUDGER_ONLY):
            if pd.judger_text:
                parts.append(f"[Judger Assessment]\n{pd.judger_text}")

        return "\n\n".join(parts)
