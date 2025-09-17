"""
LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from .base import BaseAdapter


class LoRALayer(nn.Module):
    """LoRA layer implementation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class LoRAAdapter(BaseAdapter):
    """LoRA adapter for efficient fine-tuning of LLMs."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize LoRA adapter."""
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]

        super().__init__(model_name, device, **kwargs)
        self._setup_lora_layers()
        self.freeze_base_model()

    def _setup_lora_layers(self):
        """Setup LoRA layers for target modules."""
        self.lora_layers = nn.ModuleDict()

        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.rank,
                        alpha=self.alpha,
                    ).to(self.device)
                    self.lora_layers[name] = lora_layer

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Get base model outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Apply LoRA adaptations
        adapted_states = hidden_states.clone()

        for name, lora_layer in self.lora_layers.items():
            # Find the corresponding linear layer
            module_path = name.split(".")
            current_module = self.model

            for path_part in module_path[:-1]:
                current_module = getattr(current_module, path_part)

            original_layer = getattr(current_module, module_path[-1])

            # Apply LoRA transformation
            lora_output = lora_layer(hidden_states)
            adapted_states = adapted_states + lora_output

        return adapted_states

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step for LoRA adapter."""
        self.train()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", input_ids)

        # Forward pass
        outputs = self.forward(input_ids, attention_mask)

        # Calculate loss (simple language modeling loss)
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        else:
            loss = torch.tensor(0.0, device=self.device)

        # Backward pass
        loss.backward()

        return {"loss": loss.item()}

    def save(self, path: str):
        """Save LoRA adapter weights."""
        import os

        os.makedirs(path, exist_ok=True)

        # Save LoRA weights
        torch.save(self.lora_layers.state_dict(), os.path.join(path, "lora_weights.pt"))

        # Save configuration
        config = {
            "model_name": self.model_name,
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
        }

        import json

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path: str):
        """Load LoRA adapter weights."""
        import os
        import json

        # Load configuration
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        # Update configuration
        self.rank = config["rank"]
        self.alpha = config["alpha"]
        self.target_modules = config["target_modules"]

        # Recreate LoRA layers
        self._setup_lora_layers()

        # Load weights
        weights_path = os.path.join(path, "lora_weights.pt")
        if os.path.exists(weights_path):
            self.lora_layers.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.lora_layers.parameters() if p.requires_grad)
