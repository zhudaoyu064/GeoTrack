import yaml
from src.train import train

def load_config(path='config/config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    print("Loading training configuration...")
    config = load_config()
    print("Starting training...")
    train(config)
    print("Training finished.")
