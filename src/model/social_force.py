import torch
import torch.nn as nn

class SocialForceModel(nn.Module):
    def __init__(self, input_dim):
        super(SocialForceModel, self).__init__()
        self.interaction = nn.Linear(input_dim, input_dim)

    def forward(self, positions):
        force = self.interaction(positions)
        return force
