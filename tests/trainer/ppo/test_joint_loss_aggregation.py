import torch

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_policy_loss,
    resolve_joint_loss_agg_modes,
)
from verl.utils.logger.aggregate_logger import concat_dict_to_str
from verl.utils.metric import reduce_metrics


def test_joint_loss_uses_response_equal_grpo_and_token_equal_opd_aggregation():
    # Sequence 1 has one valid token with positive advantage; sequence 2 has
    # three valid tokens with the opposite advantage.
    old_log_prob = torch.zeros((2, 3))
    log_prob = torch.zeros((2, 3))
    adv_grpo = torch.tensor([[1.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    pg_loss_grpo, _, _, _ = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=adv_grpo,
        response_mask=response_mask,
        cliprange=0.2,
        loss_agg_mode="seq-mean-token-mean",
    )
    opd_token_mean = agg_loss(
        loss_mat=-adv_grpo,
        loss_mask=response_mask,
        loss_agg_mode="token-mean",
    )

    torch.testing.assert_close(pg_loss_grpo, torch.tensor(0.0))
    torch.testing.assert_close(opd_token_mean, torch.tensor(0.5))


def test_seq_mean_token_mean_ignores_empty_sequences():
    loss = agg_loss(
        loss_mat=torch.tensor([[10.0, 10.0], [2.0, 4.0]]),
        loss_mask=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        loss_agg_mode="seq-mean-token-mean",
    )

    torch.testing.assert_close(loss, torch.tensor(3.0))


def test_joint_loss_aggregation_modes_fall_back_to_legacy_mode():
    assert resolve_joint_loss_agg_modes({"loss_agg_mode": "seq-mean-token-sum"}) == (
        "seq-mean-token-sum",
        "seq-mean-token-sum",
    )
    assert resolve_joint_loss_agg_modes(
        {
            "loss_agg_mode": "seq-mean-token-sum",
            "grpo_loss_agg_mode": "seq-mean-token-mean",
            "opd_loss_agg_mode": "token-mean",
        }
    ) == ("seq-mean-token-mean", "token-mean")


def test_string_aggregation_modes_survive_metric_reduction_and_console_formatting():
    metrics = reduce_metrics(
        {
            "actor/solver/grpo_loss_agg_mode": ["seq-mean-token-mean"] * 2,
            "actor/solver/opd_loss_agg_mode": ["token-mean"] * 2,
        }
    )

    assert metrics["actor/solver/grpo_loss_agg_mode"] == "seq-mean-token-mean"
    assert metrics["actor/solver/opd_loss_agg_mode"] == "token-mean"
    log_line = concat_dict_to_str(metrics, step=1)
    assert "actor/solver/grpo_loss_agg_mode:seq-mean-token-mean" in log_line
    assert "actor/solver/opd_loss_agg_mode:token-mean" in log_line
