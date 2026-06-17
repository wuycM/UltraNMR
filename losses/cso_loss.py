import torch
import torch.nn as nn


class CSOLoss(nn.Module):
    """
    Chemical Shift Ordering (CSO) Loss
    Used to learn the relative ordering of chemical shifts.

    It compares relative ordering within H peaks and within C peaks separately.
    """
    def __init__(self, weight=1.0, enabled=True):
        """
        Args:
            weight: Weight of the CSO loss.
            enabled: Whether to enable CSO loss.
        """
        super().__init__()
        self.weight = weight
        self.enabled = enabled
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, sequence_output, labels, mask, types, shifts, model):
        """
        Args:
            sequence_output: [B, L, d_model] Transformer encoder outputs.
            labels: [B, L] Ground-truth chemical shift values.
            mask: [B, L] Masked positions.
            types: [B, L] Atom types (0=H, 1=C).
            shifts: [B, L] Input chemical shifts.
            model: Model containing the cso_out layer.

        Returns:
            loss: CSO loss.
        """
        if not self.enabled:
            return torch.tensor(0.0, device=sequence_output.device)

        device = sequence_output.device

        masked_embeddings = sequence_output[mask]   # (num_masked, d_model)
        masked_labels = labels[mask]                # (num_masked)
        masked_types = types[mask]                  # (num_masked)

        # Get the batch index for each token.
        batch_indices = torch.arange(shifts.shape[0], device=device).unsqueeze(1).expand_as(shifts)
        masked_batch_indices = batch_indices[mask]

        all_paired_embs_list = []
        all_cso_labels_list = []

        for i in range(shifts.shape[0]):
            # Select masked samples from the current batch item.
            sample_token_indices = (masked_batch_indices == i).nonzero().squeeze(-1)
            if len(sample_token_indices) < 2:
                continue

            sample_embs = masked_embeddings[sample_token_indices]
            sample_labels = masked_labels[sample_token_indices]
            sample_types = masked_types[sample_token_indices]

            # Compare H peaks pairwise.
            h_indices = (sample_types == 0).nonzero().squeeze(-1)
            if len(h_indices) >= 2:
                h_embs = sample_embs[h_indices]
                h_labels = sample_labels[h_indices]

                # Generate all H-peak pairs.
                h_pairs_idx = torch.combinations(torch.arange(len(h_indices), device=device), r=2)
                h_embs1 = h_embs[h_pairs_idx[:, 0]]
                h_embs2 = h_embs[h_pairs_idx[:, 1]]
                h_labels1 = h_labels[h_pairs_idx[:, 0]]
                h_labels2 = h_labels[h_pairs_idx[:, 1]]

                h_paired_embs = torch.cat([h_embs1, h_embs2], dim=-1)
                h_cso_labels = (h_labels1 > h_labels2).float()

                all_paired_embs_list.append(h_paired_embs)
                all_cso_labels_list.append(h_cso_labels)

            # Compare C peaks pairwise.
            c_indices = (sample_types == 1).nonzero().squeeze(-1)
            if len(c_indices) >= 2:
                c_embs = sample_embs[c_indices]
                c_labels = sample_labels[c_indices]

                # Generate all C-peak pairs.
                c_pairs_idx = torch.combinations(torch.arange(len(c_indices), device=device), r=2)
                c_embs1 = c_embs[c_pairs_idx[:, 0]]
                c_embs2 = c_embs[c_pairs_idx[:, 1]]
                c_labels1 = c_labels[c_pairs_idx[:, 0]]
                c_labels2 = c_labels[c_pairs_idx[:, 1]]

                c_paired_embs = torch.cat([c_embs1, c_embs2], dim=-1)
                c_cso_labels = (c_labels1 > c_labels2).float()

                all_paired_embs_list.append(c_paired_embs)
                all_cso_labels_list.append(c_cso_labels)

        if not all_paired_embs_list:
            return torch.tensor(0.0, device=device)

        all_paired_embs = torch.cat(all_paired_embs_list, dim=0)
        all_cso_labels = torch.cat(all_cso_labels_list, dim=0)

        # Predict relative ordering with the model's cso_out layer.
        cso_logits = model.cso_out(all_paired_embs).squeeze(-1)
        loss = self.bce_loss(cso_logits, all_cso_labels)

        return loss  # Return the raw loss without applying the weight.
