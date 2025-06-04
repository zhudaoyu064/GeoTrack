import torch
from src.model.igvif import IGVIF

def predict(model_path, input_sequence):
    model = IGVIF(input_dim=2, latent_dim=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        output = model(input_sequence)
    return output
