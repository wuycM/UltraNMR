import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_tanimoto_similarity(fp1, fp2):
    """
    Compute the Tanimoto coefficient between two binary fingerprint matrices.

    fp1, fp2: binary fingerprint tensors of shape [N, D]
    Returns: a similarity matrix of shape [N, N]
    """
    # fp1: [N, D], fp2: [N, D]
    # Compute the intersection: sum of fp1 & fp2.
    intersection = torch.matmul(fp1, fp2.T)  # [N, N]

    # Compute the union: |fp1| + |fp2| - |fp1 & fp2|.
    fp1_sum = fp1.sum(dim=1, keepdim=True)  # [N, 1]
    fp2_sum = fp2.sum(dim=1, keepdim=True)  # [N, 1]
    union = fp1_sum + fp2_sum.T - intersection  # [N, N]

    # Tanimoto = intersection / union
    # Avoid division by zero.
    tanimoto = intersection / (union + 1e-8)
    return tanimoto


class FingerprintSimilarityLoss(nn.Module):
    """
    Contrastive loss for molecular fingerprint similarity.

    Supports both MSE and classification variants.
    The MSE variant also supports normalized Focal MSE.
    """
    def __init__(self, weight=0.1, loss_type='mse',
                 bin_size=0.05,
                 use_focal_loss=False,
                 focal_alpha=None, focal_gamma=2.0,
                 use_focal_mse=False,
                 focal_mse_gamma=2.0):
        """
        Args:
            weight: Weight of the fingerprint similarity loss.
            loss_type: 'mse' for MSE loss, 'classification' for classification loss.
            bin_size: Bin width for similarity discretization, used only when
                loss_type='classification'.
            use_focal_loss: Whether to use focal loss, only valid when
                loss_type='classification'.
            focal_alpha: Alpha parameter for focal loss.
            focal_gamma: Gamma parameter for focal loss.
            use_focal_mse: Whether to use normalized Focal MSE, only valid when
                loss_type='mse'.
            focal_mse_gamma: Gamma parameter for Focal MSE, controlling focus on
                hard examples.
        """
        super().__init__()

        from losses.focal_loss import FocalLoss

        self.weight = weight
        self.loss_type = loss_type
        self.bin_size = bin_size
        self.num_bins = int(1.0 / bin_size) + 1

        # Focal-loss configuration for classification mode.
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.ce_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        else:
            self.ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # Focal-MSE configuration for MSE mode.
        self.use_focal_mse = use_focal_mse
        self.focal_mse_gamma = focal_mse_gamma

    def tanimoto_to_bin(self, tanimoto_values):
        """
        Convert Tanimoto similarity values to bin indices.
        """
        bins = torch.clamp((tanimoto_values / self.bin_size).long(), 0, self.num_bins - 1)
        return bins

    def forward(self, sequence_output, padding_mask, fingerprints, model):
        """
        Args:
            sequence_output: [B, L, d_model] Transformer encoder outputs.
            padding_mask: [B, L] Padding mask.
            fingerprints: [B, fp_dim] Molecular fingerprints.
            model: Model containing fp_sim_classifier, only needed for
                classification mode.

        Returns:
            loss: Fingerprint similarity loss.
        """
        if fingerprints is None or self.weight <= 0:
            return torch.tensor(0.0, device=sequence_output.device)

        device = sequence_output.device
        batch_size = sequence_output.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # Compute the average pooled sequence representation using non-padding positions.
        valid_mask = ~padding_mask  # [B, L]
        valid_lengths = valid_mask.sum(dim=1, keepdim=True)  # [B, 1]

        # Sum over valid positions and then average.
        sequence_sum = (sequence_output * valid_mask.unsqueeze(-1)).sum(dim=1)  # [B, d_model]
        sequence_repr = sequence_sum / (valid_lengths.float() + 1e-8)  # [B, d_model]

        if self.loss_type == 'mse':
            # MSE loss variant.
            sequence_repr_norm = F.normalize(sequence_repr, p=2, dim=1)  # [B, d_model]

            # Compute cosine similarity.
            cosine_sim = torch.matmul(sequence_repr_norm, sequence_repr_norm.T)  # [B, B]

            # Compute Tanimoto similarity as the ground truth.
            tanimoto_sim = compute_tanimoto_similarity(fingerprints, fingerprints)  # [B, B]

            # Only use the upper triangle.
            mask_upper = torch.triu(torch.ones(batch_size, batch_size, device=device), diagonal=1).bool()

            cosine_sim_upper = cosine_sim[mask_upper]
            tanimoto_sim_upper = tanimoto_sim[mask_upper]

            if self.use_focal_mse:
                # Normalized Focal MSE
                # loss = ((1 - exp(-error)) ^ gamma) * error^2
                error = torch.abs(cosine_sim_upper - tanimoto_sim_upper)  # [N_pairs]
                focal_weight = (1.0 - torch.exp(-error)) ** self.focal_mse_gamma  # [N_pairs]
                squared_error = (cosine_sim_upper - tanimoto_sim_upper) ** 2  # [N_pairs]
                loss = (focal_weight * squared_error).mean()
            else:
                # Standard MSE loss.
                loss = F.mse_loss(cosine_sim_upper, tanimoto_sim_upper)

        elif self.loss_type == 'classification':
            # Classification loss variant.
            tanimoto_sim = compute_tanimoto_similarity(fingerprints, fingerprints)  # [B, B]

            # Only use the upper triangle.
            mask_upper = torch.triu(torch.ones(batch_size, batch_size, device=device), diagonal=1).bool()
            indices = mask_upper.nonzero(as_tuple=False)  # [N_pairs, 2]

            if indices.size(0) == 0:
                return torch.tensor(0.0, device=device)

            idx_i = indices[:, 0]  # [N_pairs]
            idx_j = indices[:, 1]  # [N_pairs]

            # Gather the corresponding embeddings.
            emb_i = sequence_repr[idx_i]  # [N_pairs, d_model]
            emb_j = sequence_repr[idx_j]  # [N_pairs, d_model]

            # Concatenate the two embeddings.
            paired_embs = torch.cat([emb_i, emb_j], dim=-1)  # [N_pairs, 2*d_model]

            # Get logits from the MLP classifier.
            fp_sim_logits = model.module.fp_sim_classifier(paired_embs)
            #fp_sim_logits = model.fp_sim_classifier(paired_embs)  # [N_pairs, num_bins]

            # Get ground-truth Tanimoto similarity and convert it to bin indices.
            tanimoto_values = tanimoto_sim[mask_upper]  # [N_pairs]
            tanimoto_bins = self.tanimoto_to_bin(tanimoto_values)  # [N_pairs]

            # Use cross-entropy loss or focal loss.
            loss = self.ce_loss_fn(fp_sim_logits, tanimoto_bins)

        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        return loss  # Return the raw loss without applying the weight.
