import torch
import torch.nn as nn

class IGVIF(nn.Module):
    def __init__(self, input_dim, latent_dim, use_multiscale=True):
        super(IGVIF, self).__init__()
        self.encoder = nn.GRU(input_dim, latent_dim, batch_first=True)
        self.decoder = nn.GRU(latent_dim, input_dim, batch_first=True)
        self.use_multiscale = use_multiscale

    def forward(self, x):
        _, h = self.encoder(x)
        decoded, _ = self.decoder(h.repeat(x.size(1), 1, 1).permute(1, 0, 2))
        return decoded
