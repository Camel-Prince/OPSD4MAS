import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_opd_grpo_joint_advantage


def test_opd_grpo_joint_uses_independent_additive_weights():
    student_log_probs = torch.tensor([[-2.0, -2.0], [-2.0, -2.0]])
    teacher_log_probs = torch.tensor([[-1.0, -3.0], [-1.5, -1.0]])
    rewards = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    mask = torch.ones_like(rewards)

    advantages, returns = compute_opd_grpo_joint_advantage(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        token_level_rewards=rewards,
        response_mask=mask,
        opd_beta=0.2,
        opd_lambda=2.0,
        grpo_outcome_weight=3.0,
        index=np.array(["prompt", "prompt"]),
        traj_index=np.array(["traj-1", "traj-2"]),
        norm_adv_by_std_in_grpo=False,
    )

    opd_component = 0.2 * 2.0 * (teacher_log_probs - student_log_probs)
    grpo_component = 3.0 * torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    expected = opd_component + grpo_component
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_opd_grpo_joint_keeps_opd_and_multiturn_grpo_masks_separate():
    student_log_probs = torch.zeros((2, 2))
    teacher_log_probs = torch.ones((2, 2))
    rewards = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    response_mask = torch.ones_like(rewards)
    grpo_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    advantages, _ = compute_opd_grpo_joint_advantage(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        token_level_rewards=rewards,
        response_mask=response_mask,
        grpo_response_mask=grpo_mask,
        opd_beta=1.0,
        grpo_outcome_weight=1.0,
        index=np.array(["prompt", "prompt"]),
        traj_index=np.array(["traj-1", "traj-2"]),
        norm_adv_by_std_in_grpo=False,
    )

    # OPD applies to both response tokens; GRPO applies only to the loss mask.
    torch.testing.assert_close(advantages, torch.tensor([[0.0, 1.0], [2.0, 1.0]]))
