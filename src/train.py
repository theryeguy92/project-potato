# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from src.utils.optim import get_optimizer_and_scheduler, get_num_training_steps
from src.utils.logging import init_logger, get_logger
import yaml
from pathlib import Path


def create_causal_mask(seq_len, device):
    """
    Create causal attention mask for language modeling.
    Shape: [seq_len, seq_len]
    Upper triangular matrix filled with -inf to prevent attending to future tokens.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    return mask


def train(model, dataloader, train_config, device="cuda"):
    """
    Main training function for the MLA LLM.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        train_config: Dictionary with training configuration
        device: Device to train on (cuda/cpu)
    """
    # Extract config values
    epochs = train_config.get('epochs', 10)
    learning_rate = train_config.get('learning_rate', 3e-4)
    weight_decay = train_config.get('weight_decay', 0.01)
    betas = tuple(train_config.get('betas', [0.9, 0.95]))
    warmup_steps = train_config.get('warmup_steps', 100)
    min_lr_ratio = train_config.get('min_lr_ratio', 0.1)
    gradient_clip = train_config.get('gradient_clip', 1.0)
    use_amp = train_config.get('use_amp', True)
    log_every_n_steps = train_config.get('log_every_n_steps', 10)
    checkpoint_dir = train_config.get('checkpoint_dir', 'checkpoints')
    save_every_epoch = train_config.get('save_every_epoch', 1)
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = init_logger(
        log_dir=train_config.get('log_dir', 'logs'),
        use_tensorboard=train_config.get('use_tensorboard', False),
        use_wandb=train_config.get('use_wandb', False),
        wandb_project=train_config.get('wandb_project', 'potato-mla'),
        wandb_run_name=train_config.get('wandb_run_name', None)
    )
    
    # Log hyperparameters
    logger.log_hyperparameters(train_config)
    
    # Calculate total training steps for scheduler
    max_steps = get_num_training_steps(dataloader, epochs)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=min_lr_ratio
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=use_amp)
    
    # Training loop
    model.train()
    global_step = 0
    
    print(f"\n{'='*80}")
    print(f"Starting training for {epochs} epochs ({max_steps} total steps)")
    print(f"{'='*80}\n")
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Create causal mask for language modeling
            seq_len = input_ids.size(1)
            causal_mask = create_causal_mask(seq_len, device)
            
            # Forward pass with mixed precision
            with autocast(enabled=use_amp):
                outputs = model(input_ids, causal_mask)
                
                # Compute language modeling loss
                # Shift so that tokens < n predict n
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                # Flatten the tokens
                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping (unscale first for accurate clipping)
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Log every N steps
            if global_step % log_every_n_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.log_metrics(
                    epoch=epoch,
                    step=global_step,
                    loss=loss.item(),
                    lr=current_lr,
                    grad_norm=grad_norm.item()
                )
        
        # End of epoch logging
        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        
        logger.log_metrics(
            epoch=epoch,
            step=global_step,
            avg_epoch_loss=avg_loss,
            lr=current_lr
        )
        
        # Save checkpoint
        if (epoch + 1) % save_every_epoch == 0:
            checkpoint_file = checkpoint_path / f"model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'global_step': global_step
            }, checkpoint_file)
            print(f"💾 Checkpoint saved: {checkpoint_file}")
    
    # Close logger
    logger.close()
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"{'='*80}\n")
    
    return model
