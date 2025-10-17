# src/utils/optim.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

def get_optimizer_and_scheduler(
    model,
    learning_rate=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.95),
    warmup_steps=100,
    max_steps=10000,
    min_lr_ratio=0.1
):
    """
    Creates AdamW optimizer and cosine learning rate scheduler with warmup.
    
    Args:
        model: The model to optimize
        learning_rate: Peak learning rate (default: 3e-4, standard for small LLMs)
        weight_decay: Weight decay coefficient (default: 0.01)
        betas: Adam beta parameters (default: (0.9, 0.95) - standard for transformers)
        warmup_steps: Number of warmup steps (default: 100)
        max_steps: Total training steps for cosine decay (default: 10000)
        min_lr_ratio: Minimum LR as ratio of peak LR (default: 0.1, so min_lr = 0.1 * learning_rate)
    
    Returns:
        optimizer, scheduler
    """
    
    # Separate parameters: apply weight decay only to weights, not biases/layer norms
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Don't apply weight decay to biases, layer norms, and embeddings
        if any(nd in name for nd in ["bias", "norm", "embed"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=1e-8
    )
    
    # Cosine learning rate scheduler with linear warmup
    def lr_lambda(current_step):
        # Warmup phase: linear increase from 0 to 1
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay phase: smooth decrease from 1 to min_lr_ratio
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(progress, 1.0)  # Cap at 1.0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def get_num_training_steps(dataloader, epochs):
    """
    Helper function to calculate total training steps.
    Useful for setting max_steps in scheduler.
    
    Args:
        dataloader: Training dataloader
        epochs: Number of training epochs
    
    Returns:
        Total number of training steps
    """
    return len(dataloader) * epochs

