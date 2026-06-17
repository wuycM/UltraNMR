"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.

Reference: LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps a linear layer with low-rank adaptation.

    Original: y = Wx + b
    LoRA: y = Wx + b + (BA)x, where B is (d_out, r) and A is (r, d_in)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of the low-rank matrices (r)
            alpha: Scaling factor (typically set to first rank value)
            dropout: Dropout probability for LoRA path
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (..., in_features)

        Returns:
            LoRA output: (..., out_features)
        """
        # x: (..., in_features)
        # lora_A: (rank, in_features)
        # lora_B: (out_features, rank)

        # Apply dropout to input
        x_dropped = self.dropout(x)

        # Compute low-rank adaptation: (BA)x
        # x @ A^T: (..., in_features) @ (in_features, rank) -> (..., rank)
        # result @ B^T: (..., rank) @ (rank, out_features) -> (..., out_features)
        lora_out = (x_dropped @ self.lora_A.T) @ self.lora_B.T

        return lora_out * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Combines a frozen linear layer with a trainable LoRA layer.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            linear: Original linear layer (will be frozen)
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout
        """
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Store dimensions for compatibility
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Freeze original linear layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    @property
    def weight(self):
        """Expose weight attribute for compatibility with PyTorch modules."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose bias attribute for compatibility with PyTorch modules."""
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original linear + LoRA adaptation.
        """
        return self.linear(x) + self.lora(x)


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.

    Args:
        model: Model to apply LoRA to
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to all Linear layers in attention and FFN.
                       Examples: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'fc1', 'fc2']

    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        # Default: apply to attention projections and FFN layers
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'out_proj',  # Attention
            'fc1', 'fc2',  # FFN
            'linear1', 'linear2',  # Alternative FFN naming
        ]

    def should_apply_lora(name: str) -> bool:
        """Check if LoRA should be applied to this module."""
        return any(target in name for target in target_modules)

    # Replace Linear layers with LinearWithLoRA
    for name, module in model.named_modules():
        # Skip if not a target module
        if not should_apply_lora(name):
            continue

        # Find parent module and attribute name
        *parent_names, attr_name = name.split('.')
        parent = model
        for parent_name in parent_names:
            parent = getattr(parent, parent_name)

        # Replace if it's a Linear layer
        if isinstance(module, nn.Linear):
            lora_layer = LinearWithLoRA(
                linear=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            setattr(parent, attr_name, lora_layer)
            print(f"  Applied LoRA to: {name}")

    return model


def apply_lora_to_transformer(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    apply_to_encoder: bool = True,
    apply_to_decoder: bool = True,
) -> nn.Module:
    """
    Apply LoRA to transformer encoder and/or decoder.

    Specifically designed for NMR2SMILES model structure.

    Args:
        model: NMR2SMILES model
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        apply_to_encoder: Whether to apply LoRA to encoder
        apply_to_decoder: Whether to apply LoRA to decoder

    Returns:
        Model with LoRA applied
    """
    print(f"Applying LoRA (rank={rank}, alpha={alpha}, dropout={dropout}):")

    # Apply to encoder
    if apply_to_encoder and hasattr(model, 'encoder'):
        print("  Applying LoRA to encoder...")
        apply_lora_to_model(
            model.encoder,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'linear1', 'linear2'],
        )

    # Apply to decoder
    if apply_to_decoder and hasattr(model, 'decoder'):
        print("  Applying LoRA to decoder...")
        apply_lora_to_model(
            model.decoder,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'linear1', 'linear2'],
        )

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable LoRA parameters.

    Args:
        model: Model with LoRA layers

    Returns:
        Number of LoRA parameters
    """
    return sum(p.numel() for p in get_lora_parameters(model))


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into the original linear layers.

    This is useful for inference to avoid the overhead of separate LoRA computation.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            # Merge: W_new = W_old + scaling * (B @ A)
            with torch.no_grad():
                lora_weight = module.lora.lora_B @ module.lora.lora_A
                lora_weight = lora_weight * module.lora.scaling
                module.linear.weight.data += lora_weight

                # Make the linear layer trainable again
                module.linear.weight.requires_grad = True
                if module.linear.bias is not None:
                    module.linear.bias.requires_grad = True

    return model
