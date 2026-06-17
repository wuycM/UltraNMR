"""
Functional Group Decoder for UltraNMR.
Uses mean pooling over encoder outputs followed by a linear classifier.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

for path in (REPO_ROOT, CURRENT_DIR):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.append(path_str)

from models.UltraNMR import UltraNMR


class FunctionalGroupDecoder(nn.Module):
    """Mean-pool encoder outputs and classify with a single linear layer."""

    def __init__(
        self,
        num_functional_groups=22,
        d_model=768,
        pretrained_encoder=None,
        freeze_encoder=False,
        use_logits=True,
    ):
        super().__init__()

        self.encoder = pretrained_encoder
        self.freeze_encoder = freeze_encoder
        self.use_logits = use_logits
        self.classifier = nn.Linear(d_model, num_functional_groups)

        if freeze_encoder and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder weights frozen")

        if not use_logits:
            self.sigmoid = nn.Sigmoid()

        print("✓ FG Decoder initialized:")
        print("  - Pooling: mean")
        print("  - Decoder: linear")
        print(f"  - Output: {'logits' if use_logits else 'probabilities'}")
        print(f"  - Num FG classes: {num_functional_groups}")

    def pool_sequence(self, encoder_output, padding_mask=None):
        """Mean-pool sequence representations."""
        if encoder_output.shape[1] == 1 and padding_mask is not None:
            padding_mask = torch.zeros(
                encoder_output.shape[0],
                1,
                dtype=torch.bool,
                device=encoder_output.device,
            )

        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()
            masked_output = encoder_output * mask
            return masked_output.sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        return encoder_output.mean(dim=1)

    def forward(self, shifts, counts, types, padding_mask):
        """Run encoder, mean-pool sequence outputs, and classify."""
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_result = self.encoder(shifts, counts, types, padding_mask)
        else:
            encoder_result = self.encoder(shifts, counts, types, padding_mask)

        if isinstance(encoder_result, tuple):
            _, _, encoder_output = encoder_result
        else:
            encoder_output = encoder_result

        pooled = self.pool_sequence(encoder_output, padding_mask)
        fg_output = self.classifier(pooled)

        if not self.use_logits:
            fg_output = self.sigmoid(fg_output)

        return fg_output

    def get_encoder_output(self, shifts, counts, types, padding_mask):
        """Get encoder output for analysis."""
        with torch.no_grad():
            _, _, encoder_output = self.encoder(shifts, counts, types, padding_mask)
        return encoder_output


def load_pretrained_encoder(checkpoint_path, config):
    """Load pretrained UltraNMR encoder."""
    print(f"\nLoading pretrained encoder from: {checkpoint_path}")

    encoder = UltraNMR(
        d_model=config.get("d_model", 768),
        nhead=config.get("nhead", 12),
        num_encoder_layers=config.get("num_encoder_layers", 16),
        dim_feedforward=config.get("dim_feedforward", 3072),
        dropout=config.get("dropout", 0.1),
        h_bin_size=config.get("h_bin_size", 0.05),
        h_max=config.get("h_max", 16.0),
        c_bin_size=config.get("c_bin_size", 1.0),
        c_max=config.get("c_max", 230.0),
        fp_sim_bin_size=config.get("fp_sim_bin_size", 0.005),
        fp_sim_mlp_hidden=config.get("fp_sim_mlp_hidden", 512),
        use_identity_embedding=config.get("use_identity_embedding", True),
        use_count_embedding=config.get("use_count_embedding", False),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder.load_state_dict(state_dict, strict=False)
    print("✓ Pretrained encoder loaded successfully")

    return encoder


def create_fg_decoder(
    checkpoint_path,
    config,
    num_functional_groups=22,
    freeze_encoder=False,
    use_logits=True,
    device="cuda",
):
    """Create functional group decoder with a pretrained or scratch UltraNMR encoder."""
    if checkpoint_path is None or checkpoint_path == "null":
        print("\n" + "=" * 60)
        print("Training from scratch (no pretrained checkpoint)")
        print("=" * 60)
        encoder = UltraNMR(
            d_model=config.get("d_model", 768),
            nhead=config.get("nhead", 12),
            num_encoder_layers=config.get("num_encoder_layers", 16),
            dim_feedforward=config.get("dim_feedforward", 3072),
            dropout=config.get("dropout", 0.1),
            h_bin_size=config.get("h_bin_size", 0.05),
            h_max=config.get("h_max", 16.0),
            c_bin_size=config.get("c_bin_size", 1.0),
            c_max=config.get("c_max", 230.0),
            fp_sim_bin_size=config.get("fp_sim_bin_size", 0.005),
            fp_sim_mlp_hidden=config.get("fp_sim_mlp_hidden", 512),
            use_identity_embedding=config.get("use_identity_embedding", True),
            use_count_embedding=config.get("use_count_embedding", False),
        )
        print("✓ Encoder initialized from scratch")
    else:
        encoder = load_pretrained_encoder(checkpoint_path, config)

    model = FunctionalGroupDecoder(
        num_functional_groups=num_functional_groups,
        d_model=config.get("d_model", 768),
        pretrained_encoder=encoder,
        freeze_encoder=freeze_encoder,
        use_logits=use_logits,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print("Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"{'=' * 60}\n")

    return model


if __name__ == "__main__":
    config = {
        "d_model": 768,
        "nhead": 12,
        "num_encoder_layers": 16,
        "dim_feedforward": 3072,
        "dropout": 0.1,
        "h_bin_size": 0.01,
        "h_max": 16.0,
        "c_bin_size": 0.1,
        "c_max": 230.0,
        "fp_sim_bin_size": 0.005,
        "fp_sim_mlp_hidden": 512,
        "use_identity_embedding": True,
        "use_count_embedding": False,
    }

    checkpoint_path = str(REPO_ROOT / "model_checkpoint" / "checkpoints_nce" / "model_epoch_1.pth")

    print("Creating FG Decoder...")
    model = create_fg_decoder(
        checkpoint_path=checkpoint_path,
        config=config,
        num_functional_groups=22,
        freeze_encoder=False,
        use_logits=True,
        device="cuda",
    )

    B, L = 4, 50
    shifts = torch.randn(B, L)
    counts = torch.ones(B, L)
    types = torch.randint(0, 2, (B, L))
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    padding_mask[:, 40:] = True

    output = model(shifts, counts, types, padding_mask)
    print(f"✓ Test forward pass successful: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
