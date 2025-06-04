import os
import zipfile

def prepare_dataset(data_dir='datasets'):
    os.makedirs(data_dir, exist_ok=True)
    # Placeholder for dataset download & unzip logic
    print(f"Preparing data in: {data_dir}")
    print("Please manually place your ETH/UCY data into this folder.")
    # Optionally automate download here if URLs are known

if __name__ == "__main__":
    prepare_dataset()
