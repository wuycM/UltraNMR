import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_soft_labels(shift_values, atom_types, h_bin_size, c_bin_size,
                                 h_max, c_max, h_num_bins, c_num_bins,
                                 sigma_h=0.01, sigma_c=0.1):
    """
    Create Gaussian soft labels for chemical shifts.

    Args:
        shift_values: [N] Raw chemical shift values in ppm.
        atom_types: [N] 0=H, 1=C
        h_bin_size, c_bin_size: Bin sizes.
        h_max, c_max: Maximum ppm values.
        h_num_bins, c_num_bins: Total number of bins.
        sigma_h, sigma_c: Standard deviation of the Gaussian distribution in ppm.

    Returns:
        h_soft_labels: [N_h, h_num_bins] Soft labels for H atoms.
        c_soft_labels: [N_c, c_num_bins] Soft labels for C atoms.
        h_mask, c_mask: Masks used for indexing.
    """
    device = shift_values.device
    N = shift_values.size(0)

    h_mask = (atom_types == 0)
    c_mask = (atom_types == 1)

    h_soft_labels = None
    c_soft_labels = None

    # Process H atoms.
    if h_mask.any():
        h_shifts = shift_values[h_mask]  # [N_h]
        N_h = h_shifts.size(0)

        # Create the center position of every bin. [h_num_bins]
        bin_centers_h = torch.arange(h_num_bins, device=device, dtype=torch.float32) * h_bin_size

        # Compute the distance from each shift to all bin centers. [N_h, h_num_bins]
        distances_h = h_shifts.unsqueeze(1) - bin_centers_h.unsqueeze(0)

        # Compute probabilities with a Gaussian: exp(-(x-mu)^2 / (2*sigma^2)).
        gaussian_h = torch.exp(-distances_h ** 2 / (2 * sigma_h ** 2))

        # Normalize so each row sums to 1.
        h_soft_labels = gaussian_h / (gaussian_h.sum(dim=1, keepdim=True) + 1e-8)

    # Process C atoms.
    if c_mask.any():
        c_shifts = shift_values[c_mask]  # [N_c]
        N_c = c_shifts.size(0)

        # Create the center position of every bin. [c_num_bins]
        bin_centers_c = torch.arange(c_num_bins, device=device, dtype=torch.float32) * c_bin_size

        # Compute the distance from each shift to all bin centers. [N_c, c_num_bins]
        distances_c = c_shifts.unsqueeze(1) - bin_centers_c.unsqueeze(0)

        # Compute probabilities with a Gaussian.
        gaussian_c = torch.exp(-distances_c ** 2 / (2 * sigma_c ** 2))

        # Normalize.
        c_soft_labels = gaussian_c / (gaussian_c.sum(dim=1, keepdim=True) + 1e-8)

    return h_soft_labels, c_soft_labels, h_mask, c_mask


class ClassificationLoss(nn.Module):
    """
    Chemical shift classification loss.
    Supports both hard-label cross entropy and Gaussian soft-label cross entropy.
    """
    def __init__(self, h_bin_size=0.05, c_bin_size=1.0,
                 h_max=16.0, c_max=230.0,
                 loss_type='hard_ce',
                 sigma_h=0.05, sigma_c=1.0,
                 use_focal_loss=False,
                 focal_alpha=None, focal_gamma=2.0):
        """
        Args:
            h_bin_size, c_bin_size: Bin sizes for H and C.
            h_max, c_max: Maximum ppm values for H and C.
            loss_type: 'hard_ce' for hard-label cross entropy, 'soft_ce' for
                Gaussian soft labels.
            sigma_h, sigma_c: Standard deviation of the Gaussian soft labels,
                used only when loss_type='soft_ce'.
            use_focal_loss: Whether to use focal loss, only valid when
                loss_type='hard_ce'.
            focal_alpha: Alpha parameter for focal loss.
            focal_gamma: Gamma parameter for focal loss.
        """
        super().__init__()

        from losses.focal_loss import FocalLoss

        self.h_bin_size = h_bin_size
        self.c_bin_size = c_bin_size
        self.h_max = h_max
        self.c_max = c_max
        self.h_num_bins = int(h_max / h_bin_size) + 1
        self.c_num_bins = int(c_max / c_bin_size) + 1

        self.loss_type = loss_type
        self.sigma_h = sigma_h
        self.sigma_c = sigma_c

        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.ce_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        else:
            self.ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def shift_to_bin(self, shift_values, atom_types):
        """
        Convert continuous chemical shift values to bin indices.
        """
        bins = torch.zeros_like(shift_values, dtype=torch.long)

        # H atoms
        h_mask = (atom_types == 0)
        if h_mask.any():
            h_shifts_ppm = shift_values[h_mask]
            h_bins = torch.clamp((h_shifts_ppm / self.h_bin_size).long(), 0, int(self.h_max / self.h_bin_size))
            bins[h_mask] = h_bins

        # C atoms
        c_mask = (atom_types == 1)
        if c_mask.any():
            c_shifts_ppm = shift_values[c_mask]
            c_bins = torch.clamp((c_shifts_ppm / self.c_bin_size).long(), 0, int(self.c_max / self.c_bin_size))
            bins[c_mask] = c_bins

        return bins

    def forward(self, h_logits, c_logits, labels, types):
        """
        Args:
            h_logits: [N, h_num_bins] Classification logits for H.
            c_logits: [N, c_num_bins] Classification logits for C.
            labels: [N] Ground-truth chemical shift values.
            types: [N] Atom types (0=H, 1=C).

        Returns:
            loss: Classification loss.
        """
        device = h_logits.device

        h_mask = (types == 0)
        c_mask = (types == 1)

        loss_h = torch.tensor(0.0, device=device)
        loss_c = torch.tensor(0.0, device=device)
        if self.loss_type == 'hard_ce':
            # Hard-label cross entropy.
            label_bins = self.shift_to_bin(labels, types)

            if h_mask.any():
                loss_h = self.ce_loss_fn(h_logits[h_mask], label_bins[h_mask])

            if c_mask.any():
                loss_c = self.ce_loss_fn(c_logits[c_mask], label_bins[c_mask])

        elif self.loss_type == 'soft_ce':
            # Gaussian soft-label cross entropy.
            h_soft_labels, c_soft_labels, _, _ = create_gaussian_soft_labels(
                labels, types,
                self.h_bin_size, self.c_bin_size,
                self.h_max, self.c_max,
                self.h_num_bins, self.c_num_bins,
                self.sigma_h, self.sigma_c
            )

            if h_mask.any():
                h_logits_filtered = h_logits[h_mask]
                h_log_probs = F.log_softmax(h_logits_filtered, dim=1)
                loss_h = -(h_soft_labels * h_log_probs).sum(dim=1).mean()

            if c_mask.any():
                c_logits_filtered = c_logits[c_mask]
                c_log_probs = F.log_softmax(c_logits_filtered, dim=1)
                loss_c = -(c_soft_labels * c_log_probs).sum(dim=1).mean()

        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        # Weighted average.
        num_h = h_mask.sum().float()
        num_c = c_mask.sum().float()
        total_samples = num_h + num_c

        if total_samples > 0:
            loss = (loss_h * num_h + loss_c * num_c) / total_samples
        else:
            loss = torch.tensor(0.0, device=device)

        return loss
