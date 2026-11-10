#!/usr/bin/env python3
"""Prepare unique DAPO-Math training and reference evaluation data for DrMAS.

Example:
    REFERENCE_DATA_DIR=./local-resource \
        python3 examples/data_preprocess/drmas_math.py
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_TRAIN_ROWS = 17_917
REFERENCE_TEST_FILES = {
    "aime24": ("AIME24/test.parquet", 30),
    "aime25": ("AIME25/test.parquet", 30),
    "amc23": ("AMC23/test.parquet", 83),
}

_LEADING_INSTRUCTION = re.compile(
    r"^Solve the following math problem step by step\.\s*"
    r"The last line of your response should be of the form Answer: \$Answer "
    r"\(without quotes\) where \$Answer is the answer to the problem\.\s*",
    re.IGNORECASE,
)
_TRAILING_INSTRUCTIONS = (
    re.compile(
        r"\s*Remember to put your answer on its own line after [\"']Answer:[\"']\.\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*Please reason step by step,\s*(?:but keep the reasoning concise\.\s*)?"
        r"(?:and )?put your final answer within\s+\\boxed\{+\}+\.\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\r?\n|\\n)*\s*Let'?s\s+think\s+step\s+by\s+step\s+and\s+output\s+"
        r"the\s+final\s+answer\s+within\s*\\boxed\{+\}+\.\s*$",
        re.IGNORECASE,
    ),
)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
    return dict(value)


def _prompt_content(prompt: Any) -> str:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, (list, tuple)) or len(prompt) != 1:
        raise ValueError(f"Expected one user message, got: {prompt!r}")
    message = _as_dict(prompt[0])
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError(f"Invalid MAS prompt message: {message!r}")
    return message["content"]


def strip_single_model_instructions(content: str) -> str:
    """Return the bare math question expected by the MAS environment."""
    question = _LEADING_INSTRUCTION.sub("", content.strip())
    for pattern in _TRAILING_INSTRUCTIONS:
        question = pattern.sub("", question).strip()
    if not question:
        raise ValueError("Prompt became empty after removing single-model instructions")
    return question


def _ground_truth(row: dict[str, Any]) -> str:
    reward_model = _as_dict(row.get("reward_model"))
    answer = reward_model.get("ground_truth")
    if answer is None:
        answer = row.get("answer", row.get("Answer"))
    if answer is None:
        raise ValueError("Missing ground-truth answer")
    return str(answer)


def _question(row: dict[str, Any]) -> str:
    if row.get("prompt") is not None:
        return strip_single_model_instructions(_prompt_content(row["prompt"]))
    question = row.get("problem", row.get("Problem"))
    if not isinstance(question, str):
        raise ValueError("Missing math question")
    return strip_single_model_instructions(question)


def _source_index(row: dict[str, Any], fallback_index: int) -> str:
    extra_info = _as_dict(row.get("extra_info"))
    source_index = extra_info.get("index")
    return str(fallback_index if source_index is None else source_index)


def make_mas_row(
    row: dict[str, Any],
    *,
    split: str,
    data_source: str,
    index: str,
) -> dict[str, Any]:
    question = _question(row)
    answer = _ground_truth(row)
    return {
        "data_source": data_source,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "prompt": [{"role": "user", "content": question}],
        "extra_info": {
            "split": split,
            "index": index,
            "answer": answer,
            "question": question,
        },
        "env_kwargs": {
            "ground_truth": answer,
            "question": question,
            "data_source": data_source,
        },
    }


def iter_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=65_536):
        yield from batch.to_pylist()


def prepare_unique_train(input_path: Path, expected_rows: int) -> list[dict[str, Any]]:
    unique_rows: dict[str, dict[str, Any]] = {}
    input_rows = 0
    for input_rows, row in enumerate(iter_parquet_rows(input_path), start=1):
        source_index = _source_index(row, input_rows - 1)
        data_source = str(row.get("data_source") or "math_dapo")
        normalized = make_mas_row(
            row,
            split="train",
            data_source=data_source,
            index=source_index,
        )
        previous = unique_rows.get(source_index)
        if previous is None:
            unique_rows[source_index] = normalized
        elif (
            previous["extra_info"]["question"] != normalized["extra_info"]["question"]
            or previous["extra_info"]["answer"] != normalized["extra_info"]["answer"]
        ):
            raise ValueError(f"Conflicting rows share extra_info.index={source_index!r}")

    if len(unique_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} unique training rows, found {len(unique_rows)} "
            f"from {input_rows} input rows"
        )
    print(f"Training data: {input_rows} input rows -> {len(unique_rows)} unique MAS rows")
    return list(unique_rows.values())


def prepare_reference_test(reference_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for data_source, (relative_path, expected_rows) in REFERENCE_TEST_FILES.items():
        path = reference_dir / relative_path
        rows = list(iter_parquet_rows(path))
        if len(rows) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows in {path}, found {len(rows)}")
        for offset, row in enumerate(rows):
            source_index = _source_index(row, offset)
            output.append(
                make_mas_row(
                    row,
                    split="test",
                    data_source=data_source,
                    index=f"{data_source}:{source_index}",
                )
            )
        print(f"Reference test {data_source}: {len(rows)} rows")

    expected_total = sum(expected for _, expected in REFERENCE_TEST_FILES.values())
    if len(output) != expected_total:
        raise AssertionError(f"Expected {expected_total} reference rows, found {len(output)}")
    return output


def validate_mas_rows(rows: list[dict[str, Any]], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    indices: set[str] = set()
    for row in rows:
        index = str(row["extra_info"]["index"])
        if index in indices:
            raise ValueError(f"Duplicate MAS row index: {index}")
        indices.add(index)
        question = row["extra_info"]["question"]
        if row["prompt"] != [{"role": "user", "content": question}]:
            raise ValueError(f"Prompt and extra_info.question differ for {index}")
        expected_env = {
            "ground_truth": row["reward_model"]["ground_truth"],
            "question": question,
            "data_source": row["data_source"],
        }
        if row["env_kwargs"] != expected_env:
            raise ValueError(f"Invalid env_kwargs for {index}")


def write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


def validate_train_file(path: Path, expected_rows: int) -> None:
    rows = list(iter_parquet_rows(path))
    validate_mas_rows(rows, expected_rows)
    print(f"Validated MAS training data: {path} ({len(rows)} unique rows)")


def validate_reference_file(path: Path) -> None:
    rows = list(iter_parquet_rows(path))
    expected_total = sum(expected for _, expected in REFERENCE_TEST_FILES.values())
    validate_mas_rows(rows, expected_total)
    counts = Counter(row["data_source"] for row in rows)
    expected_counts = {
        data_source: expected for data_source, (_, expected) in REFERENCE_TEST_FILES.items()
    }
    if counts != expected_counts:
        raise ValueError(f"Reference data-source counts differ: {dict(counts)} != {expected_counts}")
    print(f"Validated MAS reference data: {path} ({dict(counts)})")


def parse_args() -> argparse.Namespace:
    default_reference_dir = os.environ.get(
        "REFERENCE_DATA_DIR", "./local-resource"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./local-resource")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--reference_data_dir", default=default_reference_dir)
    parser.add_argument("--train_input", default=None)
    parser.add_argument("--expected_train_rows", type=int, default=EXPECTED_TRAIN_ROWS)
    parser.add_argument("--validate_train", default=None)
    parser.add_argument("--validate_reference", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_train:
        validate_train_file(Path(args.validate_train), args.expected_train_rows)
        return
    if args.validate_reference:
        validate_reference_file(Path(args.validate_reference))
        return

    reference_dir = Path(args.reference_data_dir)
    train_input = (
        Path(args.train_input)
        if args.train_input
        else reference_dir / "dapo-math-17k-processed-concise.parquet"
    )
    local_dir = Path(args.local_dir)

    train_rows = prepare_unique_train(train_input, args.expected_train_rows)
    validate_mas_rows(train_rows, args.expected_train_rows)
    write_parquet(train_rows, local_dir / "train_unique.parquet")

    test_rows = prepare_reference_test(reference_dir / "test_data")
    validate_mas_rows(test_rows, sum(expected for _, expected in REFERENCE_TEST_FILES.values()))
    write_parquet(test_rows, local_dir / "test_reference.parquet")

    stale_sampled = local_dir / "test_sampled.parquet"
    if stale_sampled.exists():
        stale_sampled.unlink()
        print(f"Removed stale evaluation dataset: {stale_sampled}")

    if args.hdfs_dir is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=str(local_dir), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
