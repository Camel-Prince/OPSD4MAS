from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.data_preprocess.drmas_math import (
    make_mas_row,
    prepare_unique_train,
    strip_single_model_instructions,
    validate_mas_rows,
)


def _source_row(index: str, question: str = "What is 1+1?", answer: str = "2"):
    return {
        "data_source": "math_dapo",
        "prompt": [
            {
                "role": "user",
                "content": (
                    question
                    + " Please reason step by step, but keep the reasoning concise. "
                    "Put your final answer within \\boxed{}."
                ),
            }
        ],
        "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": answer},
        "extra_info": {"index": index},
    }


def _write_rows(path: Path, rows) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_strip_single_model_prompt_instructions():
    assert strip_single_model_instructions(
        "What is 1+1? Please reason step by step, and put your final answer within \\boxed{{}}."
    ) == "What is 1+1?"
    assert strip_single_model_instructions(
        "What is 1+1? Please reason step by step, but keep the reasoning concise. "
        "Put your final answer within \\boxed{}."
    ) == "What is 1+1?"


def test_prepare_unique_train_deduplicates_by_uuid_not_prompt(tmp_path):
    source = tmp_path / "repeated.parquet"
    _write_rows(
        source,
        [
            _source_row("id-1"),
            _source_row("id-1"),
            _source_row("id-2"),
        ],
    )

    rows = prepare_unique_train(source, expected_rows=2)

    assert [row["extra_info"]["index"] for row in rows] == ["id-1", "id-2"]
    assert rows[0]["prompt"] == [{"role": "user", "content": "What is 1+1?"}]
    validate_mas_rows(rows, expected_rows=2)


def test_prepare_unique_train_rejects_conflicting_uuid(tmp_path):
    source = tmp_path / "conflict.parquet"
    _write_rows(source, [_source_row("same", answer="2"), _source_row("same", answer="3")])

    with pytest.raises(ValueError, match="Conflicting rows"):
        prepare_unique_train(source, expected_rows=1)


def test_make_mas_row_populates_environment_contract():
    row = make_mas_row(
        _source_row("source-id"),
        split="test",
        data_source="aime24",
        index="aime24:source-id",
    )

    assert row["ability"] == "math"
    assert row["reward_model"] == {"style": "rule", "ground_truth": "2"}
    assert row["env_kwargs"] == {
        "ground_truth": "2",
        "question": "What is 1+1?",
        "data_source": "aime24",
    }
