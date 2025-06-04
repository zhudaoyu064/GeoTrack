import torch

def rmse_loss(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

def compute_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
