import json
import time
from types import SimpleNamespace

import pytest

from verl.trainer.ppo.opd_fact_cards import (
    FactCard,
    FactArtifactStore,
    FactCardPipeline,
    FactTargetState,
    ResponseRecord,
    SlidingWindowRateLimiter,
    build_target_index,
    build_targets_for_source,
    compute_association,
    coarse_filter_cards,
    read_card_artifact,
    selected_lessons_for_target,
    select_target_facts,
    should_save_final_fact_artifact,
    should_save_trajectory_artifact,
    trajectory_advantages,
)
from verl.trainer.ppo.opd_teacher import PRIVILEGED_LESSONS_HEADER, build_fact_teacher_prompt


def record(uid, traj="t0", seq=0, agent="Solver Agent", comment=""):
    return ResponseRecord(
        response_uid=uid,
        query_uid="q0",
        trajectory_uid=traj,
        agent_id=agent,
        sequence_index=seq,
        agent_turn_index=seq,
        raw_step_index=seq,
        response_text=f"response {uid}",
        comment_last_agent=comment,
        student_prompt_messages=[{"role": "user", "content": "problem"}],
        query_text="problem",
    )


def config(**overrides):
    values = dict(
        model_name="fake", endpoint="", api_key_env="IGNORED", max_concurrent=2,
        max_retries=0, timeout=1, max_tokens=4096, max_facts_per_comment=6,
        future_window=2, extraction_records_per_request=8,
        relation_trajectories_per_request=4, relation_facts_per_request=12,
        relation_tpm_limit=110000, relation_rpm_limit=60, relation_max_concurrent=4,
        max_candidates_per_target=3, cache_dir=None, artifact_dir=None,
        save_extracted=False, save_api_audit=False,
        prompt_version="response-fact-card-v4-math-decisive", relation_prompt_version="trajectory-relation-v2",
    )
    values.update(overrides)
    return values


class FakeClient:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.prompts = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        payload = next(self.payloads)
        message = SimpleNamespace(content=json.dumps(payload))
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None)


def test_one_extracted_fact_is_one_card_with_two_stable_targets():
    records = [record("r0", seq=0), record("r1", seq=1), record("r2", seq=2, comment="use algebra")]
    client = FakeClient([
        {"records": [{"record_id": "r2", "facts": [
            {"lesson_text": "Check the algebraic substitution.", "fact_type": "correction"}
        ]}]}
    ])
    pipeline = FactCardPipeline(config(), client=client)
    cards = pipeline.extract_cards(records, global_step=4)
    assert len(cards) == 1
    assert [(target.target_response_uid, target.future_distance) for target in cards[0].targets] == [
        ("r1", 1), ("r0", 2)
    ]
    serialized = cards[0].to_dict()
    assert "exact_source_quote" not in serialized
    assert "evidence" not in serialized
    assert "association" in serialized and "relation_by_trajectory" in serialized
    assert all("association" not in target for target in serialized["targets"])


def test_terminal_response_is_also_extracted_from_its_own_body():
    records = [record("r0", seq=0), record("r1", seq=1, comment="review previous")]
    client = FakeClient([
        {"records": [
            {"record_id": "r1", "facts": [
                {"lesson_text": "Check the previous derivation.", "fact_type": "warning"}
            ]},
            {"record_id": "r1::terminal_self", "facts": [
                {"lesson_text": "Use the terminal response formula.", "fact_type": "validated_claim"}
            ]},
        ]}
    ])
    pipeline = FactCardPipeline(config(), client=client)

    cards = pipeline.extract_cards(records, global_step=4)

    assert {card.lesson_text for card in cards} == {
        "Check the previous derivation.",
        "Use the terminal response formula.",
    }
    assert {card.source_response_uid for card in cards} == {"r1"}
    assert all(
        [(target.target_response_uid, target.future_distance) for target in card.targets] == [("r0", 1)]
        for card in cards
    )
    prompt = client.prompts[0]
    assert '"record_id": "r1"' in prompt
    assert '"evaluated_response": "response r0"' in prompt
    assert '"comment": "review previous"' in prompt
    assert '"record_id": "r1::terminal_self"' in prompt
    assert '"evaluated_response": "response r1"' in prompt
    assert '"comment": ""' in prompt


def test_targets_do_not_cross_trajectory_or_include_judger():
    source = record("source", seq=3, comment="x")
    records = [record("a", seq=0), record("judge", seq=1, agent="Judger Agent"), record("b", seq=2), source,
               record("foreign", traj="other", seq=0)]
    targets = build_targets_for_source(source, records, future_window=2)
    assert [(target.target_response_uid, target.future_distance) for target in targets] == [("b", 1), ("a", 2)]


def test_targets_support_future_window_larger_than_two():
    records = [record(f"r{index}", seq=index) for index in range(5)]
    targets = build_targets_for_source(records[-1], records, future_window=4)
    assert [(target.target_response_uid, target.future_distance) for target in targets] == [
        ("r3", 1), ("r2", 2), ("r1", 3), ("r0", 4),
    ]


def test_stage_prompts_never_receive_training_signals_or_judger():
    pipeline = FactCardPipeline(config(), client=FakeClient([]))
    extraction = pipeline._extraction_prompt([{
        "record_id": "r", "problem": "P", "evaluated_response": "R", "comment": "C"
    }]).lower()
    card = FactCard("f", "q", "Do the check.", "warning", "r")
    relation = pipeline._relation_prompt([card], [{"trajectory_uid": "t", "responses": ["behavior"]}]).lower()
    for prompt in (extraction, relation):
        assert "ground_truth" not in prompt
        assert '"reward"' not in prompt
        assert '"advantage"' not in prompt
        assert "judger agent" not in prompt


def test_math_extraction_prompt_prioritizes_root_cause_and_independent_verification():
    pipeline = FactCardPipeline(config(max_facts_per_comment=2), client=FakeClient([]))
    prompt = pipeline._extraction_prompt([{
        "record_id": "r", "problem": "P", "evaluated_response": "R", "comment": "C"
    }]).lower()

    assert "comment is fallible evidence, not an authority" in prompt
    assert "decisive mathematical pivot, in both directions" in prompt
    assert "earliest invalid step" in prompt
    assert "key successful step or result" in prompt
    assert "right modeling reduction, theorem, invariant, formula" in prompt
    assert "smallest concrete step without which the conclusion would not follow" in prompt
    assert "algebraic substitutions" in prompt
    assert "incorrect formula or substitution with its correct replacement" in prompt
    assert "decisive correction of the earliest root-cause error" in prompt
    assert "decisive validated success factor or key step" in prompt
    assert "general strategy only when no more concrete mathematical lesson is available" in prompt
    assert "return at most 2 facts per record in priority order" in prompt


@pytest.mark.parametrize("payload", [{}, {"records": "bad"}, {"records": [{"record_id": "unknown", "facts": []}]}])
def test_invalid_extraction_fails_closed(payload):
    parsed = FactCardPipeline._parse_extraction(payload, {"expected"})
    assert parsed == {"expected": []}


def test_association_sign_zero_and_shared_root_value():
    advantages = trajectory_advantages({"high": 1.0, "low": 0.0})
    positive, coverage = compute_association({"high": "followed", "low": "violated"}, advantages)
    negative, _ = compute_association({"high": "violated", "low": "followed"}, advantages)
    zero, zero_coverage = compute_association({"high": "unrelated"}, advantages)
    assert positive > 0 and negative < 0 and coverage == 2
    assert zero == 0 and zero_coverage == 0
    assert trajectory_advantages({"a": 1.0, "b": 1.0}) == {"a": 0.0, "b": 0.0}


def test_target_index_and_per_target_decisions_are_independent():
    card = FactCard("f", "q", "Check units.", "warning", "source", association=0.5, targets=[
        FactTargetState("direct", 1), FactTargetState("downstream", 2)
    ])
    assert build_target_index([card]) == {"direct": [("f", 0)], "downstream": [("f", 1)]}
    card.targets[0].loo_effect_rms = 1.0
    card.targets[0].loo_net_alignment = 0.5
    card.targets[0].loo_directional_precision = 0.9
    card.targets[1].loo_effect_rms = 1.0
    card.targets[1].loo_net_alignment = -0.5
    card.targets[1].loo_directional_precision = 0.1
    select_target_facts({"f": card}, [("f", 0)], "joint_loss", 0.1, 0.6, 0.0, 2, 2, 1, -0.1)
    select_target_facts({"f": card}, [("f", 1)], "joint_loss", 0.1, 0.6, 0.0, 2, 2, 1, -0.1)
    assert card.targets[0].decision == "loo_aligned"
    assert card.targets[1].decision == "loo_rejected"
    assert card.association == 0.5


def test_rlsd_can_select_one_uncertain_while_joint_rejects_it():
    card = FactCard("f", "q", "Check units.", "warning", "source", association=0.0, targets=[
        FactTargetState("target", 1, loo_effect_rms=1.0, loo_net_alignment=0.0,
                        loo_directional_precision=0.4)
    ])
    joint = select_target_facts({"f": card}, [("f", 0)], "joint_loss", 0.1, 0.6, 0.0, 2, 2, 1, -0.1)
    assert joint == []
    card.targets[0].decision = "not_evaluated"
    rlsd = select_target_facts({"f": card}, [("f", 0)], "rlsd", 0.1, 0.6, 0.0, 2, 2, 1, -0.1)
    assert rlsd == [("f", 0)]
    assert card.targets[0].decision == "loo_uncertain"


def test_min_net_alignment_threshold_is_configurable():
    card = FactCard("f", "q", "Use the exact identity.", "validated_claim", "source", targets=[
        FactTargetState("target", 1, loo_effect_rms=1.0, loo_net_alignment=0.05,
                        loo_directional_precision=0.9)
    ])

    rejected = select_target_facts(
        {"f": card}, [("f", 0)], "joint_loss", 0.1, 0.6, 0.1, 2, 2, 1, -0.1
    )
    assert rejected == []
    assert card.targets[0].decision == "loo_rejected"

    card.targets[0].decision = "not_evaluated"
    selected = select_target_facts(
        {"f": card}, [("f", 0)], "joint_loss", 0.1, 0.6, 0.0, 2, 2, 1, -0.1
    )
    assert selected == [("f", 0)]
    assert card.targets[0].decision == "loo_aligned"


def test_selected_only_lessons_exclude_rejected_facts():
    aligned = FactCard("a", "q", "Use dimensions.", "correction", "source", targets=[
        FactTargetState("target", 1, loo_effect_rms=1.0, loo_net_alignment=0.4,
                        loo_directional_precision=0.8, decision="loo_aligned")
    ])
    rejected = FactCard("b", "q", "Guess the answer.", "strategy", "source", targets=[
        FactTargetState("target", 2, loo_effect_rms=1.0, loo_net_alignment=-0.2,
                        loo_directional_precision=0.2, decision="loo_rejected")
    ])
    cards = {"a": aligned, "b": rejected}
    assert selected_lessons_for_target(cards, [("b", 0), ("a", 0)]) == ["Use dimensions."]


def test_relation_unknown_or_duplicate_ids_fail_closed():
    payload = {"facts": [
        {"fact_id": "f", "relations": {"t": "followed"}},
        {"fact_id": "f", "relations": {"t": "violated"}},
        {"fact_id": "unknown", "relations": {"t": "followed"}},
    ]}
    assert FactCardPipeline._parse_relations(payload, {"f"}, {"t"}) == {}


def test_relation_batches_facts_and_trajectories_then_merges_all_results():
    pipeline = FactCardPipeline(config(), client=FakeClient([]))
    cards = [FactCard(f"f{index}", "q0", f"Lesson {index}.", "warning", "source") for index in range(25)]
    records = [record(f"r{index}", traj=f"t{index}") for index in range(8)]
    calls = []

    def classify_chunk(chunk, trajectories):
        calls.append((len(chunk), len(trajectories)))
        return {
            card.fact_id: {
                trajectory["trajectory_uid"]: "followed" for trajectory in trajectories
            }
            for card in chunk
        }

    pipeline._run_relation_request = classify_chunk
    pipeline.classify_relations(cards, records)

    assert sorted(calls) == sorted([(12, 4), (12, 4), (12, 4), (12, 4), (1, 4), (1, 4)])
    assert pipeline.metrics["facts/relation_requests"] == 6
    assert all(set(card.relation_by_trajectory) == {f"t{index}" for index in range(8)} for card in cards)


def test_relation_partial_chunk_failure_discards_incomplete_fact_relations():
    pipeline = FactCardPipeline(config(relation_facts_per_request=1), client=FakeClient([]))
    card = FactCard("f", "q0", "Check the identity.", "warning", "source")
    records = [record(f"r{index}", traj=f"t{index}") for index in range(8)]

    def classify_chunk(chunk, trajectories):
        if trajectories[0]["trajectory_uid"] == "t4":
            return {}
        return {"f": {trajectory["trajectory_uid"]: "followed" for trajectory in trajectories}}

    pipeline._run_relation_request = classify_chunk
    pipeline.classify_relations([card], records)

    assert card.relation_by_trajectory == {}


def test_relation_rate_limiter_reconciles_estimate_with_actual_usage():
    limiter = SlidingWindowRateLimiter(requests_per_minute=2, tokens_per_minute=100)
    reservation = limiter.acquire(40)
    limiter.reconcile(reservation, 25)
    second = limiter.acquire(75)

    assert reservation[1] == 25
    assert second[1] == 75


def test_each_api_retry_acquires_a_new_relation_rate_limit_reservation():
    class FlakyClient:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("retry")
            message = SimpleNamespace(content='{"facts": []}')
            choice = SimpleNamespace(message=message, finish_reason="stop")
            usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 17})
            return SimpleNamespace(choices=[choice], usage=usage)

    class RecordingLimiter:
        def __init__(self):
            self.acquired = []
            self.reconciled = []

        def acquire(self, estimated_tokens):
            reservation = [0.0, float(estimated_tokens)]
            self.acquired.append(reservation)
            return reservation

        def reconcile(self, reservation, actual_tokens):
            self.reconciled.append((reservation, actual_tokens))

    limiter = RecordingLimiter()
    pipeline = FactCardPipeline(config(max_retries=1), client=FlakyClient())
    result = pipeline._call_json(
        "relation", "hash", "prompt", ["f"], rate_limiter=limiter, estimated_tokens=20
    )

    assert result.payload == {"facts": []}
    assert len(limiter.acquired) == 2
    assert limiter.reconciled == [(limiter.acquired[1], 17)]


@pytest.mark.parametrize("field", [
    "relation_trajectories_per_request",
    "relation_facts_per_request",
    "relation_tpm_limit",
    "relation_rpm_limit",
    "relation_max_concurrent",
])
def test_relation_limits_must_be_positive(field):
    with pytest.raises(ValueError, match=field):
        FactCardPipeline(config(**{field: 0}), client=FakeClient([]))


def test_coarse_filter_does_not_revive_association_rejected_target():
    card = FactCard("f", "q", "Check units.", "warning", "source", association=1.0, targets=[
        FactTargetState("target", 1, decision="association_rejected")
    ])
    assert coarse_filter_cards([card]) == {"target": []}


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{item['role']}>{item['content']}</{item['role']}>" for item in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode())

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


def test_privileged_block_precedes_single_generation_marker_and_zero_budget_fails_closed():
    tokenizer = FakeTokenizer()
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "P"}]
    built = build_fact_teacher_prompt(messages, ["Check units."], tokenizer, 4, 1000, 512)
    text = tokenizer.decode(built["teacher_prompt_ids"])
    assert built["has_teacher"]
    assert text.count("<assistant>") == 1
    assert text.index(PRIVILEGED_LESSONS_HEADER) < text.index("<assistant>")
    empty = build_fact_teacher_prompt(messages, ["Check units."], tokenizer, 4, 20, 512)
    assert not empty["has_teacher"]


def test_fact_card_extracted_and_final_artifact_round_trip(tmp_path):
    card = FactCard("f", "q", "Check units.", "warning", "source", association=0.2,
                    targets=[FactTargetState("target", 1)])
    store = FactArtifactStore(str(tmp_path))
    store.write_cards("extracted", 3, [card])
    store.write_cards("final", 3, [card])
    store.flush()
    extracted = next((tmp_path / "fact_cards_extracted").iterdir())
    final = next((tmp_path / "fact_cards_final").iterdir())
    assert read_card_artifact(str(extracted))[0].to_dict() == card.to_dict()
    assert read_card_artifact(str(final))[0].to_dict() == card.to_dict()


def test_default_config_keeps_legacy_mode():
    config_text = open("verl/trainer/config/ppo_trainer.yaml", encoding="utf-8").read()
    assert "selection_mode: legacy" in config_text


def test_default_relation_batching_and_rate_limits():
    config_text = open("verl/trainer/config/ppo_trainer.yaml", encoding="utf-8").read()
    assert "relation_trajectories_per_request: 4" in config_text
    assert "relation_facts_per_request: 12" in config_text
    assert "relation_tpm_limit: 110000" in config_text
    assert "relation_rpm_limit: 60" in config_text
    assert "relation_max_concurrent: 4" in config_text


@pytest.mark.parametrize("step,total,freq,expected", [
    (1, 100, 20, False),
    (20, 100, 20, True),
    (40, 100, 20, True),
    (37, 37, 20, True),
])
def test_final_fact_artifact_schedule(step, total, freq, expected):
    assert should_save_final_fact_artifact(step, total, freq) is expected


def test_final_fact_artifact_schedule_rejects_nonpositive_frequency():
    with pytest.raises(ValueError, match="positive integer"):
        should_save_final_fact_artifact(1, 10, 0)


@pytest.mark.parametrize("step,total,freq,expected", [
    (1, 20, 10, False),
    (10, 20, 10, True),
    (19, 20, 10, False),
    (20, 20, 10, True),
])
def test_trajectory_artifact_schedule(step, total, freq, expected):
    assert should_save_trajectory_artifact(step, total, freq) is expected


def test_trajectory_artifact_schedule_rejects_nonpositive_frequency():
    with pytest.raises(ValueError, match="facts.trajectory_save_freq"):
        should_save_trajectory_artifact(1, 10, 0)


def test_default_fact_artifact_policy_is_final_only():
    config_text = open("verl/trainer/config/ppo_trainer.yaml", encoding="utf-8").read()
    assert "final_save_freq: 20" in config_text
    assert "save_extracted: False" in config_text
    assert "save_trajectories: False" in config_text
    assert "trajectory_save_freq: 1" in config_text
    assert "save_api_audit: False" in config_text


def test_inter_stage_wait_seconds_default_and_override():
    assert FactCardPipeline(config(), client=FakeClient([])).inter_stage_wait_seconds == 60.0
    assert FactCardPipeline(config(inter_stage_wait_seconds=0), client=FakeClient([])).inter_stage_wait_seconds == 0.0
    config_text = open("verl/trainer/config/ppo_trainer.yaml", encoding="utf-8").read()
    assert "inter_stage_wait_seconds: 60.0" in config_text


def test_run_waits_between_extraction_and_relation(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    pipeline = FactCardPipeline(config(inter_stage_wait_seconds=7), client=FakeClient([]))
    monkeypatch.setattr(
        pipeline, "extract_cards", lambda records, step: [FactCard("f", "q0", "Lesson.", "warning", "source")]
    )
    monkeypatch.setattr(pipeline, "classify_relations", lambda cards, records: None)
    pipeline.run([], {}, {}, global_step=1)
    assert sleeps == [7.0]
