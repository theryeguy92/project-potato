# scripts/download_dataset.py
from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env variables

# Use the path from .env or default
save_path = os.getenv('DATASET_SAVE_PATH', "data/raw/tiny_shakespeare")

dataset = load_dataset("tiny_shakespeare")
dataset.save_to_disk(save_path)
print(f"Dataset downloaded and saved to {save_path}")