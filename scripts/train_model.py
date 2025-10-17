# scripts/train_model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import numpy as np
import yaml
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from src.models.transformer import MiniLLM
from src.data.dataset import load_shakespeare
from src.train import train


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Load configurations
    print("Loading configurations...")
    with open("configs/model_config.yaml") as f:
        model_config = yaml.safe_load(f)
    
    with open("configs/training_config.yaml") as f:
        train_config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    seed = train_config.get('seed', 42)
    set_seed(seed)
    print(f"✓ Random seed set to: {seed}")
    
    # Set device
    device = train_config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    print(f"✓ Device: {device}")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # GPT2 tokenizer doesn't have a pad token by default, so we set it to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_shakespeare(
        tokenizer,
        split=train_config.get('train_split', 'train')
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.get('batch_size', 8),
        shuffle=train_config.get('shuffle', True),
        num_workers=train_config.get('num_workers', 0)
    )
    print(f"✓ Dataset loaded ({len(dataset)} samples, {len(dataloader)} batches)")
    
    # Initialize model
    print("Initializing model...")
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        latent_dim=model_config["latent_dim"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
        max_position_embeddings=model_config.get("max_position_embeddings", 2048),
        base=model_config.get("base", 10000)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Start training
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    trained_model = train(model, dataloader, train_config, device=device)
    
    print("\n✅ Training completed successfully!")
    
    return trained_model


if __name__ == "__main__":
    main()
