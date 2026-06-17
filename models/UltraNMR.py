import torch
import torch.nn as nn
from models.embedding import NmrEmbedding
from layers.feed_forward import FeedForward
from utils.scheduler import NoamScheduler

class UltraNMR(nn.Module):
    def __init__(self, d_model=768, nhead=12, num_encoder_layers=12, dim_feedforward=3072, dropout=0.1,
                 h_bin_size=0.01, h_max=16.0, c_bin_size=0.1, c_max=230.0,
                 fp_sim_bin_size=0.005, fp_sim_mlp_hidden=512,
                 use_identity_embedding=True, use_count_embedding=False,
                 lr=1e-4, weight_decay=0.0, n_warmup_steps=None):
        """
        Args:
            d_model: Embedding dimension.
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: FFN dimension.
            dropout: Dropout rate.
            h_bin_size, h_max: Bin settings for H atoms.
            c_bin_size, c_max: Bin settings for C atoms.
            fp_sim_bin_size: Bin width for fingerprint similarity discretization.
            fp_sim_mlp_hidden: Hidden dimension of the fingerprint similarity MLP classifier.
            use_identity_embedding: Whether to use identity embedding.
            use_count_embedding: Whether to use count embedding.
            lr: Learning rate.
            weight_decay: Weight decay.
            n_warmup_steps: Number of warmup steps. If set, Noam scheduler is used.
        """
        super().__init__()

        # Store optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_warmup_steps = n_warmup_steps
        self.embedding = NmrEmbedding(
            d_model,
            dropout,
            use_identity_embedding=use_identity_embedding,
            use_count_embedding=use_count_embedding
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Compute the number of bins.
        self.h_bin_size = h_bin_size
        self.c_bin_size = c_bin_size
        self.h_max = h_max
        self.c_max = c_max
        self.h_num_bins = int(h_max / h_bin_size) + 1
        self.c_num_bins = int(c_max / c_bin_size) + 1 

        # Two classification heads: one for H and one for C.
        self.h_classification_head = nn.Linear(d_model, self.h_num_bins)
        self.c_classification_head = nn.Linear(d_model, self.c_num_bins)

        self.cso_out = nn.Linear(2 * d_model, 1) #chemical shift order

        # Fingerprint similarity MLP classifier.
        # Input: 2 * d_model (concatenation of two embeddings)
        # Output: fp_sim_num_bins (number of Tanimoto similarity bins)
        self.fp_sim_bin_size = fp_sim_bin_size
        self.fp_sim_num_bins = int(1.0 / fp_sim_bin_size) + 1  # 1.0/0.05 = 20 bins

        # 3-layer MLP classifier.
        #self.fp_sim_classifier = nn.Linear(2 * d_model, self.fp_sim_num_bins)
        self.fp_sim_classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.fp_sim_num_bins)
        )

    def forward(self, shifts, counts, types, padding_mask):
        x = self.embedding(shifts, counts, types)
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        # Use different classification heads for H and C.
        h_logits = self.h_classification_head(output)  # [B, L, h_num_bins]
        c_logits = self.c_classification_head(output)  # [B, L, c_num_bins]

        return h_logits, c_logits, output

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            - optimizer only (if no warmup specified)
            - [optimizer], [lr_scheduler] (if warmup is specified)
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-6
        )

        if self.n_warmup_steps and self.n_warmup_steps > 0:
            lr_scheduler = {
                'scheduler': NoamScheduler(optimizer, self.n_warmup_steps),
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [lr_scheduler]

        return optimizer
