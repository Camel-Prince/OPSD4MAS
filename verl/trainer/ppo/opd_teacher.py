"""
OPD4MAS Teacher Prompt Construction.

Constructs teacher prompts with privileged future context for on-policy distillation.
The teacher sees the same student prompt plus future agent outputs appended,
enabling per-token KL-based advantages.

Three versions:
  V1: All future agent outputs appended verbatim.
  V2: A relevance discriminator filters future segments (see opd_relevance.py).
  V3: mem0-style memory management selects privileged context (see privileged_memory.py).
"""

import torch
import numpy as np
import re
import copy
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import os
import sys


PRIVILEGED_LESSONS_HEADER = "[Privileged Future Lessons]"


def build_fact_teacher_prompt(
    original_messages: List[Dict[str, Any]],
    selected_lessons: List[str],
    tokenizer: PreTrainedTokenizer,
    response_length: int,
    model_context_limit: int,
    privileged_token_budget: int = 512,
) -> Dict[str, Any]:
    """Rebuild a fact teacher prompt from chat messages before generation marker."""
    student_messages = copy.deepcopy(list(original_messages or []))
    student_ids = tokenizer.apply_chat_template(
        student_messages, tokenize=True, add_generation_prompt=True
    )
    result = {
        "student_prompt_ids": list(student_ids),
        "teacher_prompt_ids": list(student_ids),
        "has_teacher": False,
        "actual_privileged_tokens": 0,
        "truncated": False,
        "selected_lessons": [],
    }
    prompt_budget = max(int(model_context_limit) - int(response_length), 0)
    if not selected_lessons or prompt_budget <= 0 or not student_messages:
        return result

    lessons = [" ".join(str(lesson).strip().split()) for lesson in selected_lessons if str(lesson).strip()]
    if not lessons:
        return result

    def render(messages, active_lessons):
        teacher_messages = copy.deepcopy(messages)
        user_indices = [i for i, message in enumerate(teacher_messages) if message.get("role") == "user"]
        if not user_indices:
            return None, None
        block = PRIVILEGED_LESSONS_HEADER + "\n" + "\n".join(
            f"{index}. {lesson}" for index, lesson in enumerate(active_lessons, 1)
        )
        target = teacher_messages[user_indices[-1]]
        content = target.get("content", "")
        if not isinstance(content, str):
            return None, None
        target["content"] = content.rstrip() + "\n\n" + block
        ids = tokenizer.apply_chat_template(teacher_messages, tokenize=True, add_generation_prompt=True)
        return teacher_messages, list(ids)

    def privileged_token_count(active_lessons):
        block = PRIVILEGED_LESSONS_HEADER + "\n" + "\n".join(
            f"{index}. {lesson}" for index, lesson in enumerate(active_lessons, 1)
        )
        return len(tokenizer.encode(block, add_special_tokens=False))

    # Keep complete lessons and remove lowest-ranked ones before trimming conversation context.
    teacher_messages, teacher_ids = render(student_messages, lessons)
    while lessons and (
        teacher_ids is None
        or privileged_token_count(lessons) > privileged_token_budget
    ):
        lessons.pop()
        result["truncated"] = True
        teacher_messages, teacher_ids = render(student_messages, lessons) if lessons else (None, None)
    if not lessons or teacher_ids is None:
        return result

    # If the original context itself is too long, retain system, first problem, and recent context.
    context_messages = copy.deepcopy(student_messages)
    while len(teacher_ids) > prompt_budget and len(context_messages) > 2:
        protected = {0, len(context_messages) - 1}
        first_user = next((i for i, item in enumerate(context_messages) if item.get("role") == "user"), None)
        if first_user is not None:
            protected.add(first_user)
        removable = next((i for i in range(1, len(context_messages) - 1) if i not in protected), None)
        if removable is None:
            break
        context_messages.pop(removable)
        result["truncated"] = True
        teacher_messages, teacher_ids = render(context_messages, lessons)
    while lessons and teacher_ids is not None and len(teacher_ids) > prompt_budget:
        lessons.pop()
        result["truncated"] = True
        teacher_messages, teacher_ids = render(context_messages, lessons) if lessons else (None, None)
    if teacher_ids is None or len(teacher_ids) > prompt_budget:
        return result

    privileged_tokens = privileged_token_count(lessons)
    if privileged_tokens <= 0:
        return result
    result.update({
        "teacher_prompt_ids": teacher_ids,
        "has_teacher": True,
        "actual_privileged_tokens": privileged_tokens,
        "selected_lessons": lessons,
    })
    return result


def _print_inspect_sample(
    item_idx: int,
    record: Dict[str, Any],
    total_batch_list: List[List[Dict]],
    version: str,
) -> None:
    """Print one sample's teacher construction using pre-computed records from the main loop."""
    import os
    full = False  # always truncate; full detail is in _dump_pw_debug_json
    sep = "=" * 80
    sub_sep = "-" * 60

    def _maybe_truncate(text: str, limit: int) -> str:
        if full:
            return text
        return text[:limit] + ("..." if len(text) > limit else "")

    print(f"\n{sep}")
    print(f"  INSPECT TEACHER CONSTRUCTION — item_idx={item_idx}")
    print(f"  agent_id={record['agent_id']}  |  wg_id={record['wg_id']}  |  traj_uid={record['traj_uid']}")
    print(sep)

    # --- 1. Student input ---
    print(f"\n[1/4] STUDENT INPUT (prompt tokens={record['student_prompt_len']}, pad={record['pad_len']})")
    print(sub_sep)
    student_text = record["student_prompt_text"]
    print(_maybe_truncate(student_text, 3000))

    # --- 2. Full trajectory ---
    env_idx = record["env_idx"]
    step_idx = record["step_idx"]
    print(f"\n[2/4] FULL TRAJECTORY (env_idx={env_idx}, current_step_idx={step_idx})")
    print(sub_sep)
    if env_idx is not None:
        trajectory = total_batch_list[env_idx]
        for t_i, step_data in enumerate(trajectory):
            marker = " <<<< CURRENT" if t_i == step_idx else ""
            s_agent = step_data.get("agent_id", "?")
            s_text = step_data.get("text_response", "")
            s_active = step_data.get("active_masks", None)
            truncated = _maybe_truncate(s_text, 500)
            print(f"  [step {t_i}] agent={s_agent}  active={s_active}{marker}")
            print(f"    response: {truncated}")
            print()
    else:
        print("  WARNING: data_item not found in total_batch_list (id mismatch)")

    # --- 3. Privileged future context ---
    future_text = record["future_text"]
    print(f"\n[3/4] PRIVILEGED FUTURE CONTEXT (version={version})")
    print(sub_sep)
    if future_text:
        print(_maybe_truncate(future_text, 3000))
    else:
        print("  (empty — this is the last step, no future context)")

    # --- 4. Final teacher prompt ---
    print(f"\n[4/4] FINAL TEACHER PROMPT")
    print(sub_sep)
    teacher_text = record["teacher_prompt_text"]
    if teacher_text:
        print(f"  student_prompt_tokens={record['student_prompt_len']}")
        print(f"  suffix_tokens={record['suffix_len']}")
        print(f"  total_teacher_tokens={record['teacher_prompt_len']}")
        print(f"\n  --- Teacher full text ---")
        print(_maybe_truncate(teacher_text, 4000))
    else:
        print("  (same as student — no future context appended)")

    print(f"\n{sep}\n")


def _build_future_text_v1(
    total_batch_list: List[List[Dict]],
    env_idx: int,
    step_idx: int,
) -> str:
    """Collect all future agent outputs as privileged context (V1: full future text)."""
    future_steps = total_batch_list[env_idx][step_idx + 1:]
    if not future_steps:
        return ""
    parts = []
    for fs in future_steps:
        agent_id = fs.get("agent_id", "Unknown")
        # Skip Judger Agent — its evaluation J is injected separately via
        # _extract_judgment_from_trajectory, not as a regular future output.
        if agent_id == "Judger Agent":
            continue
        text = fs.get("text_response", "")
        if text:
            parts.append(f'The output of "{agent_id}": {text}')
    return "\n".join(parts)


def _extract_comment_from_text(text: str, tag: str = "comment_last_agent") -> str:
    """Extract content between <tag>...</tag> from agent output text.

    Args:
        text: Full agent text response.
        tag: Tag name without angle brackets.

    Returns:
        Extracted comment string, or empty string if tag not found.
    """
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _build_future_text_v4(
    total_batch_list: List[List[Dict]],
    env_idx: int,
    step_idx: int,
    comment_tag: str = "comment_last_agent",
) -> str:
    """V4: Extract only <comment_last_agent> content from future steps as privileged info.

    Unlike V1 which includes all future agent outputs verbatim, V4 only keeps
    the explicit comment each agent makes about the previous agent's output.
    """
    future_steps = total_batch_list[env_idx][step_idx + 1:]
    if not future_steps:
        return ""
    parts = []
    for fs in future_steps:
        agent_id = fs.get("agent_id", "Unknown")
        # Skip Judger Agent data — its evaluation is injected separately
        if agent_id == "Judger Agent":
            continue
        text = fs.get("text_response", "")
        comment = _extract_comment_from_text(text, comment_tag)
        if comment:
            parts.append(f'Comment from "{agent_id}": {comment}')
    return "\n".join(parts)


def _extract_judgment_from_trajectory(
    total_batch_list: List[List[Dict]],
    env_idx: int,
) -> str:
    """Extract Judger feedback J from the trajectory steps.

    Searches in reverse order for a step whose agent_id is "Judger Agent".

    Returns:
        Judgment text string, or empty string if no Judger step found.
    """
    steps = total_batch_list[env_idx]
    for step_data in reversed(steps):
        if step_data.get("agent_id") == "Judger Agent":
            text = step_data.get("text_response", "")
            if text and not text.startswith("[Judger Error"):
                return text
    return ""


def construct_teacher_data(
    total_batch_list: List[List[Dict]],
    effective_batch: List[Dict],
    tokenizers: Dict[str, PreTrainedTokenizer],
    agents_to_wg_mapping: Dict[str, str],
    max_teacher_context_length: int = 4096,
    version: str = "v1",
    use_judger: bool = False,
    use_facts: bool = False,
    fact_pools: Optional[Dict[int, List]] = None,
    fact_filter: Optional[Any] = None,
) -> List[Dict]:
    """
    For each item in effective_batch, construct teacher prompt data with privileged
    future context appended to the original student prompt.

    The teacher sees: [original_student_prompt + future_context_suffix]
    and is evaluated on the same response tokens.

    Args:
        total_batch_list: Per-env ordered list of step dicts (full trajectories).
        effective_batch: Flattened list of active step dicts (what goes into DataProto).
        tokenizers: Dict of tokenizers keyed by wg_id.
        agents_to_wg_mapping: Maps agent_id to wg_id.
        max_teacher_context_length: Max token length for teacher prompt.
        version: "v1" (full future), "v2" (relevance-filtered), "v3" (mem0-style), "v4" (comment-based).
        use_judger: Whether to inject Judger trajectory-level feedback J into privileged info.
        use_facts: Whether to use pre-extracted fact pools for privileged info.
        fact_pools: Dict mapping env_idx -> List[Fact] (pre-extracted once per trajectory).
        fact_filter: FactFilter instance for LLM relevance filtering.

    Returns:
        List of dicts, one per effective_batch item, each containing:
            'teacher_input_ids': torch.Tensor (1D, padded)
            'teacher_attention_mask': torch.Tensor (1D)
            'teacher_position_ids': torch.Tensor (1D)
    """
    env_step_index = _build_env_step_index(total_batch_list)
    _inspect_n = None  # OPD_INSPECT disabled; detail is in _dump_pw_debug_json
    _inspect_records: List[Dict[str, Any]] = []
    _fact_filtered_count = 0

    # Count how many steps will need fact filtering (for progress)
    if use_facts and fact_pools is not None and fact_filter is not None:
        _fact_total = sum(
            1 for item in effective_batch
            if fact_pools.get(env_step_index.get(id(item), (None, None))[0], [])
        )
        if _fact_total > 0:
            print(f"[Facts] filtering {_fact_total} steps (LLM per step)...",
                  file=sys.stderr, flush=True)
    else:
        _fact_total = 0

    teacher_items = []
    for item_idx, data_item in enumerate(effective_batch):
        traj_uid = data_item["traj_uid"]
        agent_id = data_item["agent_id"]
        wg_id = agents_to_wg_mapping.get(agent_id)
        if wg_id is None:
            wg_id = data_item.get("wg_id", list(tokenizers.keys())[0])
        tokenizer = tokenizers[wg_id]

        env_idx, step_idx = env_step_index.get(id(data_item), (None, None))
        if env_idx is None:
            teacher_items.append(_empty_teacher_item(data_item, tokenizer))
            if _inspect_n:
                _inspect_records.append({
                    "item_idx": item_idx, "agent_id": agent_id, "wg_id": wg_id,
                    "traj_uid": traj_uid, "env_idx": None, "step_idx": None,
                    "student_prompt_text": "(env not found)", "student_prompt_len": 0,
                    "pad_len": 0, "future_text": "", "teacher_prompt_text": "",
                    "suffix_len": 0, "teacher_prompt_len": 0,
                })
            continue

        if version == "v1":
            future_text = _build_future_text_v1(total_batch_list, env_idx, step_idx)
        elif version == "v4":
            future_text = _build_future_text_v4(total_batch_list, env_idx, step_idx)
        else:
            # v2, v3, or unknown — fall back to V1 for now
            future_text = _build_future_text_v1(total_batch_list, env_idx, step_idx)

        # --- Inject Judger trajectory-level evaluation J (改进1) ---
        if use_judger:
            judgment_text = _extract_judgment_from_trajectory(total_batch_list, env_idx)
            if judgment_text:
                if future_text:
                    future_text = f"[Judger Assessment]\n{judgment_text}\n\n{future_text}"
                else:
                    future_text = f"[Judger Assessment]\n{judgment_text}"

        # --- Fact filtering from pre-extracted pool (改进3, new design) ---
        if use_facts and fact_pools is not None and fact_filter is not None:
            fact_pool = fact_pools.get(env_idx, [])
            if fact_pool:
                try:
                    agent_role = data_item.get("agent_id", "Unknown")
                    relevant_texts = fact_filter.filter_for_step(
                        fact_pool=fact_pool,
                        current_step_idx=step_idx,
                        agent_role=agent_role,
                    )
                    _fact_filtered_count += 1
                    if _fact_total > 0 and (_fact_filtered_count % max(1, _fact_total // 5) == 0
                                            or _fact_filtered_count == _fact_total):
                        print(f"[Facts] filtering {_fact_filtered_count}/{_fact_total} steps done",
                              file=sys.stderr, flush=True)
                    if relevant_texts:
                        # Merge with existing future_text (V1/V4) if present
                        fact_text = "[Privileged Facts]\n" + "\n".join(
                            f"- {t}" for t in relevant_texts
                        )
                        if future_text:
                            future_text = fact_text + "\n\n" + future_text
                        else:
                            future_text = fact_text
                except Exception:
                    # Fall back to original future_text on filtering failure
                    pass

        if not future_text:
            teacher_items.append(_empty_teacher_item(data_item, tokenizer))
            if _inspect_n:
                _inspect_records.append({
                    "item_idx": item_idx, "agent_id": agent_id, "wg_id": wg_id,
                    "traj_uid": traj_uid, "env_idx": env_idx, "step_idx": step_idx,
                    "student_prompt_text": "(no future context)", "student_prompt_len": 0,
                    "pad_len": 0, "future_text": "", "teacher_prompt_text": "",
                    "suffix_len": 0, "teacher_prompt_len": 0,
                })
            continue

        # Use "prompts" (pure prompt without response) instead of "input_ids" (prompt+response).
        # _build_teacher_batch downstream will concatenate [teacher_prompt | response] for forward pass.
        student_prompts = data_item["prompts"]
        if isinstance(student_prompts, torch.Tensor):
            student_prompts_list = student_prompts.tolist()
        else:
            student_prompts_list = list(student_prompts)

        # Strip left-padding from prompt
        prompt_len = len(student_prompts_list)
        first_nonpad = 0
        for j in range(prompt_len):
            if student_prompts_list[j] != tokenizer.pad_token_id:
                first_nonpad = j
                break
        student_prompt_ids = student_prompts_list[first_nonpad:]

        # Tokenize the future context suffix
        suffix_text = "\n\n[Privileged Future Context]\n" + future_text
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

        # Build teacher prompt: student prompt + suffix
        teacher_prompt_ids = student_prompt_ids + suffix_ids

        # Truncate if needed (keep the beginning of student prompt + end of suffix)
        if len(teacher_prompt_ids) > max_teacher_context_length:
            # Keep student prompt intact, truncate suffix from the end
            max_suffix = max_teacher_context_length - len(student_prompt_ids)
            if max_suffix > 0:
                suffix_ids = suffix_ids[:max_suffix]
                teacher_prompt_ids = student_prompt_ids + suffix_ids
            else:
                teacher_prompt_ids = student_prompt_ids[:max_teacher_context_length]

        teacher_input_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        teacher_attn_mask = torch.ones(len(teacher_prompt_ids), dtype=torch.long)
        teacher_pos_ids = torch.arange(len(teacher_prompt_ids), dtype=torch.long)

        teacher_items.append({
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attn_mask,
            "teacher_position_ids": teacher_pos_ids,
        })

        if _inspect_n:
            _inspect_records.append({
                "item_idx": item_idx, "agent_id": agent_id, "wg_id": wg_id,
                "traj_uid": traj_uid, "env_idx": env_idx, "step_idx": step_idx,
                "student_prompt_text": tokenizer.decode(student_prompt_ids, skip_special_tokens=False),
                "student_prompt_len": len(student_prompt_ids),
                "pad_len": first_nonpad,
                "future_text": future_text,
                "teacher_prompt_text": tokenizer.decode(teacher_prompt_ids, skip_special_tokens=False),
                "suffix_len": len(suffix_ids),
                "teacher_prompt_len": len(teacher_prompt_ids),
            })

    if use_facts and _fact_filtered_count > 0:
        print(f"[Facts] filtering done: {_fact_filtered_count} steps filtered",
              file=sys.stderr, flush=True)

    return teacher_items


def _build_env_step_index(
    total_batch_list: List[List[Dict]],
) -> Dict[int, tuple]:
    """Build a mapping from data_item id -> (env_idx, step_idx) for fast lookup."""
    index = {}
    for env_idx, steps in enumerate(total_batch_list):
        for step_idx, step_data in enumerate(steps):
            index[id(step_data)] = (env_idx, step_idx)
    return index


def _empty_teacher_item(data_item: Dict, tokenizer: PreTrainedTokenizer) -> Dict:
    """Return teacher data identical to student prompt (no future context available).

    Uses "prompts" (pure prompt, no response) so that _build_teacher_batch downstream
    produces [prompt | response] — same as student, yielding advantage ≈ 0.
    """
    student_prompts = data_item["prompts"]
    if isinstance(student_prompts, torch.Tensor):
        teacher_input_ids = student_prompts.clone()
    else:
        teacher_input_ids = torch.tensor(student_prompts, dtype=torch.long)

    # Build attention mask: 0 for pad tokens (left-padded), 1 for real tokens
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    teacher_attn_mask = (teacher_input_ids != pad_token_id).long()

    teacher_pos_ids = compute_position_id_with_mask(teacher_attn_mask.unsqueeze(0)).squeeze(0)

    return {
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attn_mask,
        "teacher_position_ids": teacher_pos_ids,
    }


def pad_teacher_data(
    teacher_items: List[Dict],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Left-pad teacher prompt tensors to uniform length and stack into batch tensors.

    Returns:
        Dict with 'teacher_input_ids', 'teacher_attention_mask', 'teacher_position_ids',
        each of shape (batch_size, max_teacher_len).
    """
    max_len = max(item["teacher_input_ids"].size(0) for item in teacher_items)

    padded_input_ids = []
    padded_attn_masks = []
    padded_pos_ids = []

    for item in teacher_items:
        cur_len = item["teacher_input_ids"].size(0)
        pad_len = max_len - cur_len

        if pad_len > 0:
            pad_ids = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros(pad_len, dtype=torch.long)
            pad_pos = torch.zeros(pad_len, dtype=torch.long)

            padded_input_ids.append(torch.cat([pad_ids, item["teacher_input_ids"]]))
            padded_attn_masks.append(torch.cat([pad_mask, item["teacher_attention_mask"]]))
            padded_pos_ids.append(torch.cat([pad_pos, item["teacher_position_ids"]]))
        else:
            padded_input_ids.append(item["teacher_input_ids"])
            padded_attn_masks.append(item["teacher_attention_mask"])
            padded_pos_ids.append(item["teacher_position_ids"])

    return {
        "teacher_input_ids": torch.stack(padded_input_ids),
        "teacher_attention_mask": torch.stack(padded_attn_masks),
        "teacher_position_ids": torch.stack(padded_pos_ids),
    }
