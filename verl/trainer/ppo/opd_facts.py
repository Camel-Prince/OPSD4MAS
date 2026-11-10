"""
Fact-level privileged info extraction and filtering (Mem0-style).

Two-phase design:
  1. Extraction: after the full trajectory is collected, call an external LLM
     ONCE to extract atomic facts from the entire interaction history. Each fact
     is annotated with its source step index. The Judger's conclusion J is added
     as a "global" fact (step_idx=-1).
  2. Filtering: for each step's teacher prompt, select only facts whose
     step_idx > current_step_idx (future) or step_idx == -1 (global), then ask
     the LLM which of those are relevant to the current agent.

No caching is needed — each trajectory is extracted only once.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """A single atomic fact extracted from the trajectory."""
    text: str
    step_idx: int  # -1 for global (Judger) facts, >= 0 for per-step facts


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FACT_EXTRACTION_PROMPT = """You are analyzing a multi-agent math problem-solving trajectory.
Below is the complete interaction history, annotated with step numbers and agent roles.

Extract atomic, self-contained facts from this trajectory. Each fact must be a SINGLE
sentence expressing ONE piece of information. Label each fact with its source:
  [step N] for facts about a specific step
  [global] for facts from the Judger's overall assessment

Focus on:
- Errors found by the Verifier
- Corrections or suggestions made
- Whether the Verifier approved or rejected
- Key reasoning insights
- The Judger's overall correctness verdict and improvement suggestions

Trajectory:
{trajectory_text}

Output one fact per line in this exact format:
[step N] fact text here
[global] fact text here"""


FACT_FILTERING_PROMPT = """You are a fact selector for a multi-agent teacher-student distillation system.

The current agent is a "{agent_role}" at step {current_step} of the trajectory.
It needs privileged information from FUTURE steps to improve its current output.

Below are candidate facts extracted from future steps and the global Judger assessment.
Select ONLY the facts that would be MOST HELPFUL for the {agent_role} to know RIGHT NOW.
Consider:
- Does this fact point out an error the agent should avoid?
- Does it confirm a correct approach the agent should continue?
- Does it give strategic guidance about what to focus on?
- Is it directly relevant to the {agent_role}'s role at this step?

Return each selected fact on its own line, verbatim (exact same wording as input).
If no facts are relevant, output "NONE".

Candidate facts:
{facts_text}"""


# ---------------------------------------------------------------------------
# FactExtractor — one-shot trajectory-level extraction
# ---------------------------------------------------------------------------

class FactExtractor:
    """Extracts atomic facts from a full trajectory in a single LLM call."""

    def __init__(self, config):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=config.endpoint,
            api_key=os.environ.get(config.api_key_env, ""),
        )
        self.model = config.model_name

    def extract_all(
        self,
        trajectory_steps: List[dict],
        judger_j: str = "",
    ) -> List[Fact]:
        """Extract facts from the entire trajectory, each annotated with step index.

        Args:
            trajectory_steps: Ordered list of step dicts from total_batch_list.
                Each dict has 'agent_id' and 'text_response' keys.
            judger_j: Judger's textual assessment (becomes global fact if non-empty).

        Returns:
            List of Fact objects with step_idx annotations.
        """
        trajectory_text = self._format_trajectory(trajectory_steps, judger_j)
        if not trajectory_text.strip():
            return []

        prompt = FACT_EXTRACTION_PROMPT.format(trajectory_text=trajectory_text)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=7186,
            )
            content = resp.choices[0].message.content.strip()
            facts = self._parse_facts(content)
            return facts
        except Exception:
            return []

    def _format_trajectory(
        self,
        steps: List[dict],
        judger_j: str,
    ) -> str:
        """Format trajectory steps into a readable text for the LLM."""
        parts = []
        for i, step in enumerate(steps):
            agent_id = step.get("agent_id", "Unknown")
            text = step.get("text_response", "")
            # Skip Judger Agent entries (handled separately via judger_j)
            if agent_id == "Judger Agent":
                continue
            if text:
                parts.append(f"--- Step {i} ({agent_id}) ---\n{text}\n")
        if judger_j and not judger_j.startswith("[Judger Error"):
            parts.append(f"--- Judger Assessment (global) ---\n{judger_j}\n")
        return "\n".join(parts)

    def _parse_facts(self, raw: str) -> List[Fact]:
        """Parse LLM output lines into Fact objects."""
        facts = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r'\[step\s*(\d+)\]\s*(.+)', line, re.IGNORECASE)
            if match:
                step_idx = int(match.group(1))
                text = match.group(2).strip()
                if text:
                    facts.append(Fact(text=text, step_idx=step_idx))
                continue
            match = re.match(r'\[global\]\s*(.+)', line, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                if text:
                    facts.append(Fact(text=text, step_idx=-1))
        return facts


# ---------------------------------------------------------------------------
# FactFilter — future-step-aware LLM filtering
# ---------------------------------------------------------------------------

class FactFilter:
    """Filters facts by step index constraint + LLM relevance judgment."""

    def __init__(self, config):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=config.endpoint,
            api_key=os.environ.get(config.api_key_env, ""),
        )
        self.model = config.model_name

    def filter_for_step(
        self,
        fact_pool: List[Fact],
        current_step_idx: int,
        agent_role: str,
    ) -> List[str]:
        """Select facts relevant to the agent at current_step_idx.

        Only considers facts with step_idx > current_step_idx (future) or
        step_idx == -1 (global Judger fact). Then asks LLM to pick the
        relevant subset.

        Returns:
            List of fact text strings to inject as privileged info.
        """
        # Step-index constraint: only future facts + global facts
        candidates = [
            f for f in fact_pool
            if f.step_idx > current_step_idx or f.step_idx == -1
        ]
        if not candidates:
            return []

        # Build candidate text for LLM
        facts_text = "\n".join(
            f"[{'global' if f.step_idx == -1 else f'step {f.step_idx}'}] {f.text}"
            for f in candidates
        )

        prompt = FACT_FILTERING_PROMPT.format(
            agent_role=agent_role,
            current_step=current_step_idx,
            facts_text=facts_text,
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            content = resp.choices[0].message.content.strip()
            if content.upper() == "NONE":
                selected = []
            else:
                selected = self._parse_selected(content, candidates)
        except Exception:
            # On failure, return all candidate facts as fallback
            selected = [f.text for f in candidates]

        return selected

    def _parse_selected(
        self,
        raw: str,
        candidates: List[Fact],
    ) -> List[str]:
        """Parse LLM output back to fact texts, matching against candidates."""
        selected_texts = set()
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip any prefix the LLM might add
            line = re.sub(r'^\s*[-*•]\s*', '', line)
            line = re.sub(r'^\[(step\s*\d+|global)\]\s*', '', line, flags=re.IGNORECASE)
            if line:
                selected_texts.add(line)

        # Match back to original candidate texts (LLM might slightly reword)
        results = []
        for f in candidates:
            # Exact match
            if f.text in selected_texts:
                results.append(f.text)
            else:
                # Fuzzy: check if the core content appears
                for sel in selected_texts:
                    if len(sel) > 10 and sel[:20] in f.text:
                        results.append(f.text)
                        break
        return results
