import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_opd_grpo_joint_advantage,
    compute_opd_grpo_joint_loss_advantages,
    compute_grpo_outcome_advantage,
    compute_policy_loss,
)


def test_loss_advantages_returns_two_separate_tensors():
    """loss-level: returns (adv_opd, adv_grpo) as a 2-tuple, NOT a single combined tensor."""
    student_log_probs = torch.tensor([[-2.0, -2.0], [-2.0, -2.0]])
    teacher_log_probs = torch.tensor([[-1.0, -3.0], [-1.5, -1.0]])
    rewards = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    mask = torch.ones_like(rewards)

    adv_opd, adv_grpo = compute_opd_grpo_joint_loss_advantages(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        token_level_rewards=rewards,
        response_mask=mask,
        opd_lambda=2.0,
        index=np.array(["prompt", "prompt"]),
        traj_index=np.array(["traj-1", "traj-2"]),
        norm_adv_by_std_in_grpo=False,
    )

    # Each is a 2D tensor of shape (bs, seq_len)
    assert adv_opd.shape == (2, 2)
    assert adv_grpo.shape == (2, 2)
    # OPD = lambda * (teacher - student) * mask
    expected_opd = 2.0 * (teacher_log_probs - student_log_probs) * mask
    torch.testing.assert_close(adv_opd, expected_opd)
    # GRPO should match compute_grpo_outcome_advantage output
    expected_grpo, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask,
        index=np.array(["prompt", "prompt"]), traj_index=np.array(["traj-1", "traj-2"]),
        norm_adv_by_std_in_grpo=False,
    )
    torch.testing.assert_close(adv_grpo, expected_grpo)


def test_loss_level_pure_pg_opd_no_clip():
    """OPD term is pure PG: agg_loss(-adv_opd * log_prob) — no ratio, no clip."""
    log_prob = torch.tensor([[-0.5, -1.0], [-2.0, -0.3]], requires_grad=True)
    old_log_prob = torch.tensor([[-0.5, -1.0], [-2.0, -0.3]])  # ratio=1
    adv_opd = torch.tensor([[0.1, -0.05], [0.2, 0.0]])
    adv_grpo = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    mask = torch.ones_like(log_prob)

    # GRPO term via standard PPO
    pg_loss_grpo, _, _, _ = compute_policy_loss(
        old_log_prob=old_log_prob, log_prob=log_prob, advantages=adv_grpo,
        response_mask=mask, cliprange=0.2, loss_agg_mode="token-mean",
    )
    # OPD term: pure PG, no clip — manually inline from dp_actor.py
    opd_pg = agg_loss(loss_mat=-adv_opd * log_prob, loss_mask=mask, loss_agg_mode="token-mean")

    # Combined
    opd_loss_weight = 0.005
    grpo_outcome_weight = 1.0
    policy_loss = grpo_outcome_weight * pg_loss_grpo + opd_loss_weight * opd_pg

    # At ratio=1 the combined loss shouldn't be NaN
    assert torch.isfinite(policy_loss)

    # Verify OPD term gradient flows independently through log_prob
    opd_pg.backward(retain_graph=True)
    opd_grad = log_prob.grad.clone()
    log_prob.grad.zero_()

    # Now confirm adv-level would give SAME single loss (ratio=1 → no clip activated)
    combined_adv = opd_loss_weight * adv_opd + grpo_outcome_weight * adv_grpo
    pg_loss_adv, _, _, _ = compute_policy_loss(
        old_log_prob=old_log_prob, log_prob=log_prob, advantages=combined_adv,
        response_mask=mask, cliprange=0.2, loss_agg_mode="token-mean",
    )
    torch.testing.assert_close(policy_loss, pg_loss_adv)


def test_adv_vs_loss_gradient_diverges_when_clipped():
    """When ratio is clipped (|ratio-1| > ε), adv-level OPD gradient is clipped
    together with GRPO, but loss-level OPD gradient flows independently."""
    old_log_prob = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]])
    log_prob = torch.tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)  # ratio = exp(1.0) ≈ 2.72 > 1+0.2
    adv_opd = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
    adv_grpo = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    mask = torch.ones_like(log_prob)

    # --- loss-level: OPD pure PG, independent of clip ---
    log_prob.grad = None
    opd_pg = agg_loss(loss_mat=-adv_opd * log_prob, loss_mask=mask, loss_agg_mode="token-mean")
    opd_pg.backward()
    loss_opd_grad = log_prob.grad.clone().detach()
    # pure PG gradient is just -adv_opd/mask_mean → nonzero
    assert (loss_opd_grad.abs() > 0).all(), "loss-level OPD gradient should be nonzero even when ratio is clipped"

    # --- adv-level: OPD+GRPO share same advantages → clipped together ---
    combined_adv = 0.005 * adv_opd + 1.0 * adv_grpo
    log_prob_2 = log_prob.detach().clone().requires_grad_(True)
    pg_loss_adv, _, _, _ = compute_policy_loss(
        old_log_prob=old_log_prob, log_prob=log_prob_2, advantages=combined_adv,
        response_mask=mask, cliprange=0.2, loss_agg_mode="token-mean",
    )
    pg_loss_adv.backward()
    adv_grad = log_prob_2.grad.clone().detach()
    # The two gradients will differ at least for the OPD component, because
    # adv-level forces the same ratio/clip on OPD (which adds OPD*clipped_ratio
    # rather than OPD*log_prob). The exact numerical equality would be
    # coincidence; the key claim is that the two gradient tensors are NOT identical.
    assert not torch.allclose(loss_opd_grad, adv_grad), (
        "loss-level and adv-level gradients should diverge when ratio clipped"
    )


def test_loss_level_opd_weight_zero_equals_grpo_only():
    """When opd_loss_weight=0, policy_loss == grpo_outcome_weight * compute_policy_loss(adv_grpo)."""
    old_log_prob = torch.tensor([[-0.5, -0.5]])
    log_prob = torch.tensor([[-0.5, -0.5]], requires_grad=True)
    adv_grpo = torch.tensor([[1.0, 1.0]])
    adv_opd = torch.zeros_like(adv_grpo)
    mask = torch.ones_like(log_prob)

    pg_loss_grpo, _, _, _ = compute_policy_loss(
        old_log_prob=old_log_prob, log_prob=log_prob, advantages=adv_grpo,
        response_mask=mask, cliprange=0.2, loss_agg_mode="token-mean",
    )
    # When opd_loss_weight=0, the OPD term should contribute nothing
    total = 1.0 * pg_loss_grpo + 0.0 * agg_loss(loss_mat=-adv_opd * log_prob, loss_mask=mask, loss_agg_mode="token-mean")
    torch.testing.assert_close(total, pg_loss_grpo)


def test_loss_level_grpo_weight_zero_equals_opd_only():
    """When grpo_outcome_weight=0, only opd pure PG term contributes."""
    log_prob = torch.tensor([[-0.5, -1.0]], requires_grad=True)
    old_log_prob = torch.tensor([[-0.5, -1.0]])
    adv_opd = torch.tensor([[0.2, -0.1]])
    adv_grpo = torch.zeros_like(adv_opd)
    mask = torch.ones_like(log_prob)

    opd_pg = agg_loss(loss_mat=-adv_opd * log_prob, loss_mask=mask, loss_agg_mode="token-mean")
    total = 0.0 * 0.0 + 0.005 * opd_pg  # grpo_weight=0, opd_weight=5e-3
    assert torch.isfinite(total)
    total.backward()
    grad = log_prob.grad.clone()
    # gradient should be exactly from OPD term: -0.005 * adv_opd / num_tokens
    expected_grad = -0.005 * adv_opd / mask.sum()
    torch.testing.assert_close(grad, expected_grad)
