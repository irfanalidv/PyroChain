"""
Main adapter class that wraps different adapter types.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from tqdm import tqdm

from .base import BaseAdapter
from .lora import LoRAAdapter


class LightweightAdapter:
    """
    Main adapter class that provides a unified interface for different adapter types.
    
    Supports LoRA and other lightweight adaptation methods for efficient fine-tuning
    of large language models on feature engineering tasks.
    """
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        adapter_type: str = "lora",
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize the lightweight adapter."""
        self.model_name = model_name
        self.device = device
        self.adapter_type = adapter_type
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules
        
        # Initialize the specific adapter
        if adapter_type.lower() == "lora":
            self.adapter = LoRAAdapter(
                model_name=model_name,
                device=device,
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")
        
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapter."""
        return self.adapter.forward(input_ids, attention_mask)
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings for text."""
        return self.adapter.get_embeddings(text)
    
    def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text using the tokenizer."""
        return self.adapter.encode_text(text)
    
    def train(
        self,
        training_data: List[Dict[str, Any]],
        task_description: str,
        epochs: int = 3,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        validation_split: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the adapter on task-specific data.
        
        Args:
            training_data: List of training samples
            task_description: Description of the training task
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training metrics and results
        """
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.adapter.lora_layers.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )
        
        # Prepare data
        train_data, val_data = self._split_data(training_data, validation_split)
        train_loader = self._create_data_loader(train_data, batch_size)
        val_loader = self._create_data_loader(val_data, batch_size) if val_data else None
        
        # Training loop
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, task_description)
            
            # Validation phase
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate_epoch(val_loader, task_description)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                **val_metrics
            }
            training_history.append(epoch_metrics)
            
            # Save best model
            if val_metrics and val_metrics.get("val_loss", float('inf')) < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                # Could save checkpoint here
            
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics.get('val_loss', 'N/A'):.4f}")
        
        self.training_history = training_history
        
        return {
            "training_history": training_history,
            "best_val_loss": best_val_loss,
            "total_parameters": self.adapter.get_trainable_parameters(),
            "adapter_type": self.adapter_type
        }
    
    def _split_data(self, data: List[Dict[str, Any]], validation_split: float) -> tuple:
        """Split data into training and validation sets."""
        import random
        random.shuffle(data)
        
        split_idx = int(len(data) * (1 - validation_split))
        return data[:split_idx], data[split_idx:]
    
    def _create_data_loader(self, data: List[Dict[str, Any]], batch_size: int):
        """Create a data loader for the given data."""
        # Simple batching implementation
        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batches.append(self._prepare_batch(batch_data))
        return batches
    
    def _prepare_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of data for training."""
        texts = [sample.get("text", "") for sample in batch_data]
        
        # Encode all texts
        encodings = [self.adapter.encode_text(text) for text in texts]
        
        # Stack tensors
        input_ids = torch.stack([enc["input_ids"].squeeze(0) for enc in encodings])
        attention_mask = torch.stack([enc["attention_mask"].squeeze(0) for enc in encodings])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For language modeling
        }
    
    def _train_epoch(self, data_loader, task_description: str) -> Dict[str, float]:
        """Train for one epoch."""
        self.adapter.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(data_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Forward pass
            metrics = self.adapter.train_step(batch)
            loss = metrics["loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.adapter.lora_layers.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {"loss": total_loss / num_batches}
    
    def _validate_epoch(self, data_loader, task_description: str) -> Dict[str, float]:
        """Validate for one epoch."""
        self.adapter.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validation"):
                metrics = self.adapter.train_step(batch)
                total_loss += metrics["loss"]
                num_batches += 1
        
        return {"val_loss": total_loss / num_batches}
    
    def save(self, path: Union[str, Path]):
        """Save the adapter."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save adapter weights
        self.adapter.save(str(path))
        
        # Save training history
        if self.training_history:
            import json
            with open(path / "training_history.json", "w") as f:
                json.dump(self.training_history, f, indent=2)
    
    def load(self, path: Union[str, Path]):
        """Load the adapter."""
        path = Path(path)
        
        # Load adapter weights
        self.adapter.load(str(path))
        
        # Load training history if available
        history_path = path / "training_history.json"
        if history_path.exists():
            import json
            with open(history_path, "r") as f:
                self.training_history = json.load(f)
