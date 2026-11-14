
"""Reusable PyTorch training engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    epochs: int = 100
    device: str = "auto"
    gradient_clip: Optional[float] = None
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    log_interval: int = 10


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


def resolve_device(device: str = "auto") -> torch.device:
    """Select an available computation device."""

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    selected = torch.device(device)

    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")

    return selected


class Trainer:
    """General-purpose supervised neural network trainer."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Optional[TrainerConfig] = None,
        scheduler=None,
    ):
        self.config = config or TrainerConfig()

        if self.config.epochs < 1:
            raise ValueError("epochs must be positive")

        self.device = resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.history = TrainingHistory()
        self.best_val_loss = float("inf")

    def _run_batch(self, inputs, targets, training=True):
        """Execute one forward pass and optional backward pass."""

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if training:
                loss.backward()

                if self.config.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                self.optimizer.step()

        return loss.item(), inputs.size(0)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train the model for one epoch."""

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            loss, batch_size = self._run_batch(
                inputs, targets, training=True
            )

            total_loss += loss * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Training loader is empty.")

        return total_loss / total_samples

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Calculate average validation loss."""

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for inputs, targets in val_loader:
            loss, batch_size = self._run_batch(
                inputs, targets, training=False
            )

            total_loss += loss * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Validation loader is empty.")

        return total_loss / total_samples

    def save_checkpoint(self, epoch: int, filename: str):
        """Save model and optimizer state."""

        output_dir = Path(self.config.checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": {
                "train_loss": self.history.train_loss,
                "val_loss": self.history.val_loss,
                "learning_rates": self.history.learning_rates,
            },
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = (
                self.scheduler.state_dict()
            )

        torch.save(checkpoint, output_dir / filename)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> TrainingHistory:
        """Run the complete training process."""

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history.train_loss.append(train_loss)

            val_loss = None

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history.val_loss.append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    if self.config.save_best:
                        self.save_checkpoint(
                            epoch, "best_model.pt"
                        )

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(
                        val_loss if val_loss is not None
                        else train_loss
                    )
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(lr)

            if (
                epoch == 1
                or epoch % self.config.log_interval == 0
                or epoch == self.config.epochs
            ):
                message = (
                    f"Epoch {epoch}/{self.config.epochs} "
                    f"| Train Loss: {train_loss:.6f}"
                )

                if val_loss is not None:
                    message += f" | Val Loss: {val_loss:.6f}"

                message += f" | LR: {lr:.6g}"
                print(message)

        return self.history

