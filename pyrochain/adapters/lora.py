"""
LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from .base import BaseAdapter

logger = logging.getLogger(__name__)


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
        target_modules: list[str] | None = None,
        **kwargs,
    ):
        """Initialize LoRA adapter."""
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]

        super().__init__(model_name, device, **kwargs)
        self._setup_lora_layers()
        self.freeze_base_model()

    def _hidden_size(self) -> int:
        return int(self.model.get_input_embeddings().weight.shape[-1])

    def _setup_lora_layers(self):
        """Setup LoRA layers.

        One low-rank residual layer is created per target ``nn.Linear`` module whose input and
        output size equal the model's hidden size; each is applied to the final hidden states.
        If no target module qualifies (e.g. GPT-2 uses ``Conv1D``), a single residual layer on
        the final hidden states is created so the adapter always has trainable parameters.
        """
        self.lora_layers = nn.ModuleDict()
        hidden = self._hidden_size()

        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules) and (
                isinstance(module, nn.Linear)
                and module.in_features == hidden
                and module.out_features == hidden
            ):
                # ModuleDict keys may not contain "."
                self.lora_layers[name.replace(".", "__")] = LoRALayer(
                    in_features=hidden, out_features=hidden, rank=self.rank, alpha=self.alpha
                )

        if len(self.lora_layers) == 0:
            self.lora_layers["final_hidden"] = LoRALayer(
                in_features=hidden, out_features=hidden, rank=self.rank, alpha=self.alpha
            )
        self.lora_layers.to(device=self.device, dtype=self.model.dtype)
        # New modules start in training mode (dropout on); follow the base model's mode so
        # embeddings after load() are deterministic.
        self.lora_layers.train(self.model.training)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass: base-model hidden states plus the low-rank residual adaptations."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._adapt(outputs.last_hidden_state)

    def _adapt(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add every LoRA layer's low-rank residual to the base hidden states."""
        adapted_states = hidden_states
        for lora_layer in self.lora_layers.values():
            adapted_states = adapted_states + lora_layer(hidden_states)
        return adapted_states

    def get_embeddings(self, text: str) -> torch.Tensor:
        """Mean-pooled embedding of ``text`` from the *adapted* hidden states.

        The base class pools the frozen base model's output; using it here meant a trained
        adapter had no effect on extracted features. Because ``lora_B`` starts at zero, an
        untrained adapter returns exactly the base-model embedding.
        """
        encoding = self.encode_text(text)
        mask = encoding["attention_mask"].unsqueeze(-1).to(self.model.dtype)
        with torch.no_grad():
            hidden = self._adapt(self.model(**encoding).last_hidden_state)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the next-token language-modelling loss for one batch.

        Logits are obtained by projecting adapted hidden states onto the (tied) input embedding
        matrix, as in GPT-2's LM head. Returns ``{"loss": tensor}``; the caller runs backward.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", input_ids)

        hidden = self.forward(input_ids, attention_mask)
        logits = hidden @ self.model.get_input_embeddings().weight.T

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return {"loss": loss}

    def save(self, path: str | Path) -> None:
        """Save LoRA weights (``lora_weights.pt``) and configuration (``config.json``)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.lora_layers.state_dict(), path / "lora_weights.pt")
        config = {
            "model_name": self.model_name,
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load LoRA weights saved by :meth:`save`.

        Raises:
            FileNotFoundError: If ``config.json`` or ``lora_weights.pt`` is missing.
        """
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        weights_path = path / "lora_weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")
        if config.get("model_name") != self.model_name:
            logger.warning(
                "Adapter in %s was trained on %r but is being loaded into %r",
                path,
                config.get("model_name"),
                self.model_name,
            )

        self.rank = config["rank"]
        self.alpha = config["alpha"]
        self.target_modules = config["target_modules"]
        self._setup_lora_layers()
        self.lora_layers.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.lora_layers.parameters() if p.requires_grad)
