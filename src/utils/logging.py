# src/utils/logging.py
import sys
import time
from datetime import datetime
from pathlib import Path

# Optional tensorboard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Optional wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainingLogger:
    """
    Unified logger for training metrics with support for console, tensorboard, and wandb.
    """
    
    def __init__(
        self,
        log_dir="logs",
        use_tensorboard=False,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None
    ):
        """
        Initialize the training logger.
        
        Args:
            log_dir: Directory for logs (used by tensorboard)
            use_tensorboard: Enable tensorboard logging
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
            wandb_run_name: W&B run name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console logging setup
        self.start_time = time.time()
        
        # Tensorboard setup
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        if self.use_tensorboard:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_log_dir = self.log_dir / f"tensorboard_{timestamp}"
            self.writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"📊 Tensorboard logging enabled: {tb_log_dir}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            print("⚠️  Tensorboard requested but not installed. Install with: pip install tensorboard")
        
        # Wandb setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "potato-mla",
                name=wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            print(f"🔗 Weights & Biases logging enabled: {wandb.run.url}")
        elif use_wandb and not WANDB_AVAILABLE:
            print("⚠️  Wandb requested but not installed. Install with: pip install wandb")
        
        # Create text log file
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._write_to_file(f"Training started at {datetime.now()}\n")
        self._write_to_file("=" * 80 + "\n")
    
    def _write_to_file(self, message):
        """Write message to log file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def _format_time(self, seconds):
        """Format elapsed time as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def log_metrics(self, epoch, step=None, **metrics):
        """
        Log training metrics to all enabled backends.
        
        Args:
            epoch: Current epoch number
            step: Current global step (optional)
            **metrics: Arbitrary keyword arguments for metrics (loss, lr, etc.)
        
        Example:
            logger.log_metrics(epoch=1, step=100, loss=2.5, lr=3e-4, perplexity=12.2)
        """
        elapsed = time.time() - self.start_time
        
        # Console logging
        metrics_str = " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                                   for k, v in metrics.items()])
        step_str = f" Step {step}" if step is not None else ""
        console_msg = f"[{self._format_time(elapsed)}] Epoch {epoch}{step_str} | {metrics_str}"
        print(console_msg)
        
        # File logging
        self._write_to_file(console_msg + "\n")
        
        # Tensorboard logging
        if self.use_tensorboard and self.writer is not None:
            global_step = step if step is not None else epoch
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, global_step)
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {"epoch": epoch, **metrics}
            if step is not None:
                log_dict["step"] = step
            wandb.log(log_dict)
    
    def log_hyperparameters(self, config):
        """
        Log hyperparameters/configuration.
        
        Args:
            config: Dictionary of hyperparameters
        """
        print("\n" + "=" * 80)
        print("HYPERPARAMETERS")
        print("=" * 80)
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")
        
        # Write to file
        self._write_to_file("\nHyperparameters:\n")
        for key, value in config.items():
            self._write_to_file(f"  {key}: {value}\n")
        self._write_to_file("\n")
        
        # Log to wandb
        if self.use_wandb:
            wandb.config.update(config)
    
    def close(self):
        """Close all logging backends."""
        elapsed = time.time() - self.start_time
        end_msg = f"\nTraining completed in {self._format_time(elapsed)}"
        print(end_msg)
        self._write_to_file(end_msg + "\n")
        
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()
        
        if self.use_wandb:
            wandb.finish()


# Global logger instance
_global_logger = None


def init_logger(log_dir="logs", use_tensorboard=False, use_wandb=False, 
                wandb_project=None, wandb_run_name=None):
    """
    Initialize the global logger instance.
    
    Args:
        log_dir: Directory for logs
        use_tensorboard: Enable tensorboard logging
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
    
    Returns:
        TrainingLogger instance
    """
    global _global_logger
    _global_logger = TrainingLogger(
        log_dir=log_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name
    )
    return _global_logger


def log_metrics(epoch, loss, lr=None, step=None, **kwargs):
    """
    Convenience function for logging metrics. Compatible with existing train.py usage.
    
    Args:
        epoch: Current epoch
        loss: Training loss
        lr: Learning rate (optional)
        step: Global step (optional)
        **kwargs: Additional metrics
    """
    if _global_logger is None:
        # Fallback: simple console logging if logger not initialized
        print(f"Epoch {epoch} | Loss: {loss:.6f}" + 
              (f" | LR: {lr:.2e}" if lr is not None else ""))
        return
    
    metrics = {"loss": loss}
    if lr is not None:
        metrics["lr"] = lr
    metrics.update(kwargs)
    
    _global_logger.log_metrics(epoch=epoch, step=step, **metrics)


def get_logger():
    """Get the global logger instance."""
    return _global_logger

