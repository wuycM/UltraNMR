"""
Isomer NCE Loss Module

NCE Loss for distinguishing isomers based on NMR embeddings and fingerprints.
- Positive: f(y) where y is the fingerprint of the anchor molecule
- Negative: f(y_i) for other isomers with the same molecular formula
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    """
    2-layer MLP for projection
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IsomerNCELoss(nn.Module):
    """
    NCE Loss for Isomer Discrimination

    L = -log(exp(sim(z, f(y+))) / (exp(sim(z, f(y+))) + sum(exp(sim(z, f(y-)))))

    where:
    - z = g(mean_pool(nmr_embedding)) : NMR embedding projected through MLP g
    - f(y) : fingerprint projected through MLP f
    - y+ : positive fingerprint (same molecule)
    - y- : negative fingerprints (other isomers with same formula)
    """

    def __init__(
        self,
        d_model: int = 768,
        fp_dim: int = 2048,
        proj_hidden_dim: int = 512,
        proj_output_dim: int = 256,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: NMR embedding dimension
            fp_dim: fingerprint dimension (2048 for Morgan)
            proj_hidden_dim: hidden dimension for projection MLPs
            proj_output_dim: output dimension for projection MLPs
            temperature: NCE temperature
            dropout: dropout rate for MLPs
        """
        super().__init__()

        self.temperature = temperature

        # Projection MLP for NMR embedding: g(z)
        self.nmr_projector = ProjectionMLP(
            input_dim=d_model,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            dropout=dropout
        )

        # Projection MLP for fingerprint: f(y)
        self.fp_projector = ProjectionMLP(
            input_dim=fp_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            dropout=dropout
        )

    def forward(
        self,
        sequence_output: torch.Tensor,  # (B, L, d_model)
        padding_mask: torch.Tensor,  # (B, L)
        positive_fps: torch.Tensor,  # (B, 2048)
        negative_fps: torch.Tensor,  # (B, num_neg, 2048)
    ) -> torch.Tensor:
        """
        Compute NCE loss

        Args:
            sequence_output: Transformer encoder output (B, L, d_model)
            padding_mask: Padding mask (B, L), True for padding positions
            positive_fps: Positive fingerprints (B, 2048)
            negative_fps: Negative fingerprints (B, num_neg, 2048)

        Returns:
            NCE loss scalar
        """
        batch_size = sequence_output.shape[0]
        num_negatives = negative_fps.shape[1]

        # Mean pooling over non-padding positions
        # mask: True for valid positions
        valid_mask = ~padding_mask  # (B, L)
        valid_mask = valid_mask.unsqueeze(-1).float()  # (B, L, 1)

        # Masked mean pooling
        masked_output = sequence_output * valid_mask  # (B, L, d_model)
        sum_output = masked_output.sum(dim=1)  # (B, d_model)
        count = valid_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        nmr_embedding = sum_output / count  # (B, d_model)

        # Project NMR embedding: z = g(nmr_embedding)
        z = self.nmr_projector(nmr_embedding)  # (B, proj_dim)
        z = F.normalize(z, dim=-1)  # L2 normalize

        # Project positive fingerprint: f(y+)
        pos_proj = self.fp_projector(positive_fps)  # (B, proj_dim)
        pos_proj = F.normalize(pos_proj, dim=-1)

        # Project negative fingerprints: f(y-)
        # Reshape for batch processing
        neg_fps_flat = negative_fps.view(-1, negative_fps.shape[-1])  # (B*num_neg, 2048)
        neg_proj_flat = self.fp_projector(neg_fps_flat)  # (B*num_neg, proj_dim)
        neg_proj = neg_proj_flat.view(batch_size, num_negatives, -1)  # (B, num_neg, proj_dim)
        neg_proj = F.normalize(neg_proj, dim=-1)

        # Compute similarities
        # Positive similarity: (B,)
        pos_sim = (z * pos_proj).sum(dim=-1) / self.temperature

        # Negative similarities: (B, num_neg)
        neg_sim = torch.bmm(neg_proj, z.unsqueeze(-1)).squeeze(-1) / self.temperature

        # NCE Loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(exp(pos) + sum(exp(neg)))
        # = -pos + logsumexp([pos, neg1, neg2, ...])
        all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, 1 + num_neg)
        loss = -pos_sim + torch.logsumexp(all_sim, dim=-1)

        return loss.mean()

    def get_nmr_embedding(
        self,
        sequence_output: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get projected NMR embedding (for inference/evaluation)
        """
        valid_mask = ~padding_mask
        valid_mask = valid_mask.unsqueeze(-1).float()

        masked_output = sequence_output * valid_mask
        sum_output = masked_output.sum(dim=1)
        count = valid_mask.sum(dim=1).clamp(min=1)
        nmr_embedding = sum_output / count

        z = self.nmr_projector(nmr_embedding)
        z = F.normalize(z, dim=-1)

        return z

    def get_fp_embedding(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """
        Get projected fingerprint embedding (for inference/evaluation)
        """
        fp_proj = self.fp_projector(fingerprints)
        fp_proj = F.normalize(fp_proj, dim=-1)
        return fp_proj
