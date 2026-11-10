"""Lossless validation-row export and trajectory-level pass@1/pass@4 aggregation."""
import json
from collections import defaultdict
from pathlib import Path


def plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def append_validation_records(directory, batch, input_texts, output_texts, scores):
    # input_texts is the pre-expansion query batch; per-response input IDs are authoritative.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = batch.non_tensor_batch
    count = len(output_texts)
    with (directory / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
        for i in range(count):
            row = {k: plain(meta[k][i]) for k in (
                'uid', 'traj_uid', 'agent_id', 'wg_id', 'data_source', 'pass', 'extra_info'
            ) if k in meta}
            row.update(response=output_texts[i], response_reward=float(scores[i]),
                       response_token_ids=plain(batch.batch['responses'][i]))
            if 'prompts' in batch.batch:
                row['prompt_token_ids'] = plain(batch.batch['prompts'][i])
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def summarize(path, expected_questions=None, repeats=4):
    groups = defaultdict(dict)
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            key = (str(r['data_source']), str(r['uid']))
            trajectory = str(r['traj_uid'])
            outcome = r['pass']
            if outcome not in (True, False, 0, 1, 0.0, 1.0):
                raise ValueError('Non-binary trajectory outcome')
            if trajectory in groups[key] and groups[key][trajectory] != bool(outcome):
                raise ValueError('Inconsistent duplicated trajectory outcome')
            groups[key][trajectory] = bool(outcome)
    if not groups:
        raise ValueError('No saved trajectories')
    if expected_questions is not None and len(groups) != expected_questions:
        raise ValueError(f'Expected {expected_questions} questions, got {len(groups)}')
    if any(len(v) != repeats for v in groups.values()):
        raise ValueError(f'Every question must have exactly {repeats} distinct complete trajectories')
    p1 = sum(sum(v.values()) / repeats for v in groups.values()) / len(groups)
    p4 = sum(any(v.values()) for v in groups.values()) / len(groups)
    return {'questions':len(groups), 'complete_trajectories':len(groups)*repeats,
            'repeats':repeats, 'pass@1':p1, 'pass@4':p4,
            'scorer':'retained DrMAS math grader; Minerva requires scoring qualification'}


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions', type=Path)
    p.add_argument('--questions', type=int)
    a=p.parse_args()
    print(json.dumps(summarize(a.predictions,a.questions),indent=2))
