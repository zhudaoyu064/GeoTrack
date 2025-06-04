import torch

def compute_ade(pred_traj, true_traj):
    """
    Average Displacement Error (ADE)
    """
    return torch.norm(pred_traj - true_traj, dim=-1).mean()

def compute_fde(pred_traj, true_traj):
    """
    Final Displacement Error (FDE)
    """
    return torch.norm(pred_traj[:, -1] - true_traj[:, -1], dim=-1).mean()

def compute_collision_rate(pred_traj, min_dist=0.2):
    """
    Calculates the proportion of predicted trajectories with collisions.
    """
    batch_size, seq_len, _ = pred_traj.shape
    collision_count = 0

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            dist = torch.norm(pred_traj[i] - pred_traj[j], dim=-1)
            if (dist < min_dist).any():
                collision_count += 1

    return collision_count / (batch_size * (batch_size - 1) / 2 + 1e-6)
