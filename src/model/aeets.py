import torch

class AEETS:
    def __init__(self, entropy_threshold=0.05, exploration_rate=0.3):
        self.entropy_threshold = entropy_threshold
        self.exploration_rate = exploration_rate

    def adjust_learning(self, loss_entropy):
        if loss_entropy > self.entropy_threshold:
            mode = "exploration"
        else:
            mode = "exploitation"
        return mode
