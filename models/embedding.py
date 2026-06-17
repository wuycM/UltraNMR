import torch
import torch.nn as nn
from layers.fourier_features import FourierFeatures
from layers.feed_forward import FeedForward

class NmrEmbedding(nn.Module):
    def __init__(
        self,
        d_model=512,
        dropout=0.1,
        strategy='lin_float_int',     # Default strategy: lin_float_int
        h_x_min=0.01, h_x_max=16.0,
        c_x_min=0.01, c_x_max=230.0,
        num_freqs_per_type=None,      # If None, compute automatically
        funcs='both',                 # 'both' | 'sin' | 'cos'
        trainable=False,
        sigma=10,
        use_identity_embedding=False,  # Whether to use identity embedding (negative values go directly into Fourier)
        use_count_embedding=True,      # Whether to use count embedding
    ):
        super().__init__()
        self.d_model = d_model
        self.use_identity_embedding = use_identity_embedding
        self.use_count_embedding = use_count_embedding

        # ====== Create Fourier feature generators ======
        self.h_fourier = FourierFeatures(
            strategy=strategy, x_min=h_x_min, x_max=h_x_max,
            trainable=trainable, funcs=funcs, sigma=sigma,
            num_freqs=num_freqs_per_type or max(8, d_model // 4)
        )
        self.c_fourier = FourierFeatures(
            strategy=strategy, x_min=c_x_min, x_max=c_x_max,
            trainable=trainable, funcs=funcs, sigma=sigma,
            num_freqs=num_freqs_per_type or max(8, d_model // 4)
        )

        # Note: the two Fourier output dimensions may differ here.
        h_out_dim = self.h_fourier.num_features()
        c_out_dim = self.c_fourier.num_features()

        # ====== Build separate MLPs for H and C and align them to d_model ======
        self.h_shift_ffn = FeedForward(in_dim=h_out_dim, out_dim=d_model, depth=5)
        self.c_shift_ffn = FeedForward(in_dim=c_out_dim, out_dim=d_model, depth=5)

        # H atoms can optionally include an additional count feature.
        if use_count_embedding:
            self.h_count_ffn = FeedForward(in_dim=1, out_dim=d_model, depth=1)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)


    def forward(self, shifts, counts, types):
        """
        shifts: [B, L] NMR shift values. Negative values may indicate masked identities,
            such as -1, -2, -3, and so on.
        counts: [B, L] Peak areas, atom counts, or similar count features.
        types:  [B, L]   0=H, 1=C

        When use_identity_embedding=True, masked positions in shifts are negative
        values (-1, -2, -3, ...). These values are fed directly into the Fourier
        features for encoding.
        """
        B, L = shifts.shape

        # ====== Use shifts directly, including negative values, as Fourier inputs ======
        shifts_unsqueezed = shifts.unsqueeze(-1)  # [B, L, 1]

        # ====== Compute separate Fourier encodings, which can handle negative values ======
        h_fourier_out = self.h_fourier(shifts_unsqueezed)  # [B, L, F_h]
        c_fourier_out = self.c_fourier(shifts_unsqueezed)  # [B, L, F_c]

        # ====== Map each encoding to the same dimension with separate MLPs ======
        h_embed = self.h_shift_ffn(h_fourier_out)  # [B, L, d_model]
        c_embed = self.c_shift_ffn(c_fourier_out)  # [B, L, d_model]

        # ====== Select the corresponding embedding based on atom type ======
        is_h_mask = (types == 0).unsqueeze(-1)  # [B, L, 1]
        shift_embedding = torch.where(is_h_mask, h_embed, c_embed)  # [B, L, d_model]

        # ====== Add count features when enabled; only H atoms use them ======
        if self.use_count_embedding:
            count_embedding = self.h_count_ffn(counts.unsqueeze(-1))     # [B, L, d_model]
            count_embedding = count_embedding * is_h_mask.float()        # Keep count features only for H atoms
            out = self.layer_norm(shift_embedding + count_embedding)
        else:
            out = self.layer_norm(shift_embedding)

        return self.dropout(out)
