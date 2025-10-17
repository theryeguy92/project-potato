# scripts/generate_text.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
from transformers import GPT2Tokenizer
from pathlib import Path
import argparse

from src.models.transformer import MiniLLM


def load_model_from_checkpoint(checkpoint_path, model_config, device='cuda'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_config: Model configuration dictionary
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    # Initialize tokenizer to get vocab size
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Initialize model
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full checkpoint and state_dict only
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        print(f"✓ Loaded checkpoint from epoch {epoch} (loss: {loss})")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded checkpoint (state dict only)")
    
    model.eval()
    return model


def create_causal_mask(seq_len, device):
    """Create causal attention mask for generation."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    return mask


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt="",
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    sampling_method="top_p",
    device='cuda'
):
    """
    Generate text from a prompt using various sampling strategies.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens for sampling
        top_p: Keep top tokens with cumulative probability p (nucleus sampling)
        sampling_method: 'greedy', 'temperature', 'top_k', 'top_p'
        device: Device to run on
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    if prompt:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    else:
        # Start with BOS token if no prompt
        input_ids = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device)
    
    # Generate tokens
    for _ in range(max_new_tokens):
        # Get model predictions
        seq_len = input_ids.size(1)
        causal_mask = create_causal_mask(seq_len, device)
        
        logits = model(input_ids, causal_mask)
        
        # Get logits for the last token
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature != 1.0 and sampling_method != 'greedy':
            next_token_logits = next_token_logits / temperature
        
        # Sample next token based on method
        if sampling_method == 'greedy':
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        elif sampling_method == 'temperature':
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        elif sampling_method == 'top_k':
            # Top-k sampling
            top_k_values, top_k_indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
            # Zero out logits not in top-k
            next_token_logits_filtered = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits_filtered.scatter_(1, top_k_indices, top_k_values)
            probs = torch.softmax(next_token_logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        elif sampling_method == 'top_p':
            # Nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            # Zero out removed indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def interactive_mode(model, tokenizer, device, temperature=1.0, sampling_method='top_p'):
    """
    Interactive text generation mode.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        device: Device to run on
        temperature: Sampling temperature
        sampling_method: Sampling method to use
    """
    print("\n" + "="*80)
    print("INTERACTIVE TEXT GENERATION MODE")
    print("="*80)
    print(f"Settings: method={sampling_method}, temperature={temperature}")
    print("Commands:")
    print("  - Type your prompt and press Enter to generate")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'help' for settings info")
    print("="*80 + "\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if prompt.lower() == 'help':
                print(f"\nCurrent settings:")
                print(f"  Sampling method: {sampling_method}")
                print(f"  Temperature: {temperature}")
                print(f"  Max tokens: 100")
                print()
                continue
            
            if not prompt:
                print("⚠️  Please enter a prompt or 'quit' to exit\n")
                continue
            
            print("\nGenerating...\n")
            
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=100,
                temperature=temperature,
                sampling_method=sampling_method,
                device=device
            )
            
            print("Generated text:")
            print("-" * 80)
            print(generated)
            print("-" * 80 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained MLA LLM")
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (e.g., checkpoints/model_epoch_9.pt)'
    )
    
    # Generation settings
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt for generation (if not provided, enters interactive mode)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate (default: 100)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature - higher = more random (default: 1.0)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['greedy', 'temperature', 'top_k', 'top_p'],
        default='top_p',
        help='Sampling method (default: top_p)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k value for top_k sampling (default: 50)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p value for nucleus sampling (default: 0.9)'
    )
    
    # Device settings
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Load configurations
    print("Loading configurations...")
    with open("configs/model_config.yaml") as f:
        model_config = yaml.safe_load(f)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                for cp in checkpoints:
                    print(f"  - {cp}")
            else:
                print("  (no checkpoints found)")
        else:
            print("  (checkpoints directory doesn't exist)")
        return
    
    model = load_model_from_checkpoint(args.checkpoint, model_config, args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded ({total_params:,} parameters)\n")
    
    # Generate text
    if args.prompt is not None:
        # Single generation mode
        print("="*80)
        print("TEXT GENERATION")
        print("="*80)
        print(f"Prompt: {args.prompt}")
        print(f"Method: {args.method}, Temperature: {args.temperature}")
        print("="*80 + "\n")
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sampling_method=args.method,
            device=args.device
        )
        
        print("Generated text:")
        print("-" * 80)
        print(generated)
        print("-" * 80)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args.device, args.temperature, args.method)


if __name__ == "__main__":
    main()

