import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, weight=0.1, temperature=0.07, use_fully_vectorized=True):
        """
        InfoNCE loss for H-C peak alignment.

        Args:
            weight: Loss weight
            temperature: Temperature for InfoNCE loss
            use_fully_vectorized: If True, use fully vectorized implementation (no batch loop).
                                  If False, use original implementation with batch loop.
        """
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.use_fully_vectorized = use_fully_vectorized

    def forward(self, sequence_output, types, mask, padding_mask,
                h_c_shifts, c_shifts_list, shifts):

        if h_c_shifts is None or c_shifts_list is None or self.weight <= 0:
            return torch.tensor(0.0, device=sequence_output.device)

        if self.use_fully_vectorized:
            return self._forward_fully_vectorized(sequence_output, types, mask, padding_mask,
                                                   h_c_shifts, c_shifts_list, shifts)
        else:
            return self._forward_original(sequence_output, types, mask, padding_mask,
                                          h_c_shifts, c_shifts_list, shifts)

    def _forward_original(self, sequence_output, types, mask, padding_mask,
                          h_c_shifts, c_shifts_list, shifts):
        """Original implementation with batch loop."""
        device = sequence_output.device
        h_global = (types == 0) & (~padding_mask)

        pair_losses = []

        for i in range(shifts.shape[0]):
            sample_h = h_global[i]
            if not sample_h.any():
                continue

            h_indices = sample_h.nonzero().squeeze(-1)
            if len(h_indices) == 0:
                continue

            h_embeddings = sequence_output[i, h_indices]
            h_true_c_shifts = h_c_shifts[i, :len(h_indices)]

            sample_c_shifts = c_shifts_list[i].to(device)

            if len(sample_c_shifts) == 0:
                continue

            c_mask_sample = (types[i] == 1) & (~padding_mask[i])
            if not c_mask_sample.any():
                continue

            c_indices_sample = c_mask_sample.nonzero().squeeze(-1)
            c_embeddings = sequence_output[i, c_indices_sample]

            # Calculate cosine similarity matrix once: [num_h, num_c]
            h_embeddings_norm = F.normalize(h_embeddings, p=2, dim=1)
            c_embeddings_norm = F.normalize(c_embeddings, p=2, dim=1)
            cosine_sim_matrix = torch.matmul(h_embeddings_norm, c_embeddings_norm.T) / self.temperature

            # Filter out H peaks with zero true_c_shift
            valid_h_mask = torch.abs(h_true_c_shifts) >= 1e-6  # [num_h]
            if not valid_h_mask.any():
                continue

            # Get valid H peaks and their true C shifts
            valid_h_true_c_shifts = h_true_c_shifts[valid_h_mask]  # [num_valid_h]
            valid_cosine_sim = cosine_sim_matrix[valid_h_mask]  # [num_valid_h, num_c]

            # Compute distance matrix: [num_valid_h, num_c]
            distances = torch.abs(valid_h_true_c_shifts.unsqueeze(1) - sample_c_shifts.unsqueeze(0))

            # Find closest C peak for each H peak: [num_valid_h]
            pos_indices = torch.argmin(distances, dim=1)
            # Each H peak is matched to its unique closest C peak.
            valid_logits = valid_cosine_sim  # [num_valid_h, num_c]
            valid_targets = pos_indices  # [num_valid_h]

            sample_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
            pair_losses.append(sample_loss)

        if pair_losses:
            return torch.cat(pair_losses).mean()
        else:
            return torch.tensor(0.0, device=device)

    def _forward_fully_vectorized(self, sequence_output, types, mask, padding_mask,
                                   h_c_shifts, c_shifts_list, shifts):
        """Fully vectorized implementation without batch loop."""
        device = sequence_output.device
        batch_size = shifts.shape[0]

        # Global masks: [batch_size, seq_len]
        h_mask = (types == 0) & (~padding_mask)
        c_mask = (types == 1) & (~padding_mask)

        # 1. Extract all H peaks across the batch
        h_batch_indices, h_seq_indices = h_mask.nonzero(as_tuple=True)  # [total_h]

        if len(h_batch_indices) == 0:
            return torch.tensor(0.0, device=device)

        # All H embeddings: [total_h, d_model]
        h_embeddings = sequence_output[h_batch_indices, h_seq_indices]

        # Get true C shift for each H peak
        # Compute cumulative sum to get relative indices within each sample
        h_counts_cumsum = torch.cat([
            torch.tensor([0], device=device),
            h_mask.sum(dim=1).cumsum(dim=0)[:-1]
        ])  # [batch_size]
        h_relative_indices = torch.arange(len(h_batch_indices), device=device) - h_counts_cumsum[h_batch_indices]
        h_true_c_shifts = h_c_shifts[h_batch_indices, h_relative_indices]  # [total_h]

        # Filter out H peaks with zero true_c_shift
        valid_h_mask = torch.abs(h_true_c_shifts) >= 1e-6
        if not valid_h_mask.any():
            return torch.tensor(0.0, device=device)

        h_batch_indices = h_batch_indices[valid_h_mask]  # [total_valid_h]
        h_embeddings = h_embeddings[valid_h_mask]  # [total_valid_h, d_model]
        h_true_c_shifts = h_true_c_shifts[valid_h_mask]  # [total_valid_h]

        # 2. Extract all C peaks across the batch
        c_batch_indices, c_seq_indices = c_mask.nonzero(as_tuple=True)  # [total_c]

        if len(c_batch_indices) == 0:
            return torch.tensor(0.0, device=device)

        # All C embeddings: [total_c, d_model]
        c_embeddings = sequence_output[c_batch_indices, c_seq_indices]

        # Get C shift values from c_shifts_list
        # Pad and concatenate all C shifts
        c_shifts_padded = torch.nn.utils.rnn.pad_sequence(
            [c_shifts_list[i].to(device) for i in range(batch_size)],
            batch_first=True, padding_value=0.0
        )  # [batch_size, max_c_len]
        c_counts_cumsum = torch.cat([
            torch.tensor([0], device=device),
            c_mask.sum(dim=1).cumsum(dim=0)[:-1]
        ])
        c_relative_indices = torch.arange(len(c_batch_indices), device=device) - c_counts_cumsum[c_batch_indices]
        c_shift_values = c_shifts_padded[c_batch_indices, c_relative_indices]  # [total_c]

        # 3. Compute full cosine similarity matrix: [total_valid_h, total_c]
        h_norm = F.normalize(h_embeddings, p=2, dim=1)
        c_norm = F.normalize(c_embeddings, p=2, dim=1)
        full_cosine_sim = torch.matmul(h_norm, c_norm.T) / self.temperature

        # 4. Create same-sample mask: only H-C pairs from the same sample can be matched
        # [total_valid_h, total_c]
        same_sample_mask = h_batch_indices.unsqueeze(1) == c_batch_indices.unsqueeze(0)

        # 5. Compute distance matrix: [total_valid_h, total_c]
        distances = torch.abs(h_true_c_shifts.unsqueeze(1) - c_shift_values.unsqueeze(0))

        # Set distances between different samples to infinity
        distances = distances.masked_fill(~same_sample_mask, float('inf'))

        # 6. Find the closest C peak for each H peak
        pos_indices = distances.argmin(dim=1)  # [total_valid_h]

        # 7. Get valid logits and targets
        valid_logits = full_cosine_sim  # [total_valid_h, total_c]
        valid_same_sample_mask = same_sample_mask  # [total_valid_h, total_c]

        # Mask out logits for C peaks from different samples (set to -inf)
        valid_logits = valid_logits.masked_fill(~valid_same_sample_mask, float('-inf'))

        valid_targets = pos_indices  # [total_valid_h]

        # 8. Compute cross-entropy loss
        loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')

        return loss
