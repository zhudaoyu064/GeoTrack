import torch
from src.inference import predict

if __name__ == "__main__":
    model_path = "experiments/checkpoints/best_model.pth"
    test_sequence = torch.randn(1, 8, 2)  # Replace with actual input
    print("Loading model and running inference...")
    output = predict(model_path, test_sequence)
    print("Output:", output)
