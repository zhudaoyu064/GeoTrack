import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2

class PedestrianDataset(Dataset):
    def __init__(self, data_path, seq_len=8, pred_len=12, transform=None):
        self.data = pd.read_csv(data_path)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transform = transform

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        past = self.data.iloc[idx:idx + self.seq_len].values
        future = self.data.iloc[idx + self.seq_len:idx + self.seq_len + self.pred_len].values
        if self.transform:
            past, future = self.transform(past, future)
        return torch.tensor(past, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)

def get_dataloader(data_path, batch_size, shuffle=True, num_workers=4):
    dataset = PedestrianDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
