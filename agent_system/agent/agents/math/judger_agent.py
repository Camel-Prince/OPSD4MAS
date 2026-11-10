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

"""Judger Agent — GT-based trajectory-level text evaluation for OPD privileged info.

After the MAS (Solver-Verifier) loop completes, the Judger reads the full
trajectory and ground-truth answer, then produces a concise textual evaluation J.
This J is injected into every step's teacher privileged context during OPD
teacher prompt construction.
"""

import os
import sys
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents.base import BaseAgent


JUDGER_PROMPT = """You are a trajectory evaluator for a multi-agent math problem-solving system.
Below is the full interaction trajectory of a Solver-Verifier team working on a math problem.

[PROBLEM]
{problem}

[GROUND TRUTH ANSWER]
{ground_truth}

[FULL TRAJECTORY]
{trajectory}

Judge ONLY whether the trajectory's final answer is CORRECT or INCORRECT
compared to the ground truth. Reply in ONE short sentence:
- If correct: "CORRECT." optionally followed by a brief note on what worked
  (without restating the answer).
- If incorrect: "INCORRECT." optionally followed by a brief note on where the
  reasoning went wrong (WITHOUT stating or hinting at the correct answer).
Do NOT reveal, restate, or hint at the ground-truth answer itself.
Do NOT include any XML tags.
"""


@AgentRegistry.register("Judger Agent")
class JudgerAgent(BaseAgent):
    """Post-hoc trajectory evaluator using an external LLM API.

    Unlike Solver/Verifier agents, the Judger does NOT use Ray worker groups
    or call() — it invokes an external API (e.g., DeepSeek) via the OpenAI client.
    """

    def __init__(self, wg_id: str = "external", tokenizer=None, processor=None, config=None):
        super().__init__(
            "Judger Agent", JUDGER_PROMPT,
            wg_id=wg_id,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )
        judger_cfg = config.algorithm.opd.judger
        self.client = OpenAI(
            base_url=judger_cfg.endpoint,
            api_key=os.environ.get(judger_cfg.api_key_env, ""),
        )
        self.model_name = judger_cfg.model_name
        self.max_tokens = judger_cfg.max_tokens
        self.temperature = judger_cfg.temperature
        self.max_retries = judger_cfg.max_retries
        self.timeout = judger_cfg.timeout
        self.max_concurrent = judger_cfg.max_concurrent
        # Per-batch records (trajectory_idx -> {input, output}); consumed by
        # _dump_pw_debug_json so the Judger's full input/output are preserved
        # in the JSON dump even though its batch is excluded from training.
        self.debug_records: List[dict] = []

    def call(self, *args, **kwargs):
        """JudgerAgent does not use the standard call() interface.

        Use judge() or judge_batch() instead.
        """
        raise NotImplementedError(
            "JudgerAgent uses judge() directly for external API calls, not call()."
        )

    def judge(self, trajectory_text: str, problem: str, ground_truth: str,
              record_idx: int = -1) -> str:
        """Evaluate a single trajectory against ground truth.

        Args:
            trajectory_text: The full team_context (accumulated agent outputs).
            problem: The math problem description.
            ground_truth: The correct answer.
            record_idx: If >= 0, store I/O in self.debug_records for JSON dump.

        Returns:
            Textual evaluation string, or an error marker on failure.
        """
        prompt = self.prompt.format(
            problem=problem,
            ground_truth=ground_truth,
            trajectory=trajectory_text,
        )
        # Detail dump is handled by _dump_pw_debug_json; keep terminal clean.
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                result = resp.choices[0].message.content.strip()
                if record_idx >= 0:
                    self.debug_records.append({
                        "idx": record_idx,
                        "input": prompt,
                        "output": result,
                    })
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    err_msg = f"[Judger Error: {e}]"
                    if record_idx >= 0:
                        self.debug_records.append({
                            "idx": record_idx,
                            "input": prompt,
                            "output": err_msg,
                        })
                    return err_msg
        return ""

    def judge_batch(
        self,
        trajectories: List[str],
        problems: List[str],
        ground_truths: List[str],
    ) -> List[str]:
        """Evaluate a batch of trajectories in parallel via ThreadPoolExecutor.

        Args:
            trajectories: Per-sample trajectory texts.
            problems: Per-sample problem descriptions.
            ground_truths: Per-sample ground-truth answers.

        Returns:
            List of judgment strings (same length as input), with error markers
            for failed API calls.
        """
        active_count = sum(1 for t, p in zip(trajectories, problems) if t and p)
        print(f"[Judger] starting batch: {active_count} trajectories, "
              f"concurrency={self.max_concurrent}", file=sys.stderr, flush=True)

        results = [""] * len(trajectories)
        completed = 0
        self.debug_records.clear()
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = {}
            for i, (traj, prob, gt) in enumerate(
                zip(trajectories, problems, ground_truths)
            ):
                if traj and prob:
                    futures[executor.submit(self.judge, traj, prob, gt, i)] = i
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[Judger Error: {e}]"
                completed += 1
                if completed % max(1, active_count // 10) == 0 or completed == active_count:
                    print(f"[Judger] {completed}/{active_count} done", file=sys.stderr, flush=True)

        print(f"[Judger] batch complete: {active_count} trajectories", file=sys.stderr, flush=True)
        return results