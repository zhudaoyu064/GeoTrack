import torch
from src.data_loader import get_dataloader
from src.model.igvif import IGVIF
from src.model.utils import rmse_loss

def train(config):
    dataloader = get_dataloader(config["dataset_path"], config["batch_size"])
    model = IGVIF(input_dim=2, latent_dim=config["latent_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        for past, future in dataloader:
            pred = model(past)
            loss = rmse_loss(pred, future)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
