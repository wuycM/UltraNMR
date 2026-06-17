import torch
import torch.nn as nn

from losses.classification_loss import ClassificationLoss
from losses.cso_loss import CSOLoss
from losses.fingerprint_loss import FingerprintSimilarityLoss
from losses.infonce_loss import InfoNCELoss


class NMRLoss(nn.Module):
    """
    Composite loss function including:
    1. Classification Loss: chemical shift classification loss
    2. CSO Loss: chemical shift ordering loss
    3. Fingerprint Similarity Loss: molecular fingerprint similarity loss
    4. InfoNCE Loss: H-C correspondence learning loss
    """
    def __init__(self,
                 # Classification loss arguments
                 h_bin_size=0.05, c_bin_size=1.0,
                 h_max=16.0, c_max=230.0,
                 loss_type='hard_ce',
                 sigma_h=0.05, sigma_c=1.0,
                 use_focal_loss=False,
                 focal_alpha=None, focal_gamma=2.0,
                 # CSO loss arguments
                 cso_weight=1.0,
                 use_cso_loss=True,
                 # Fingerprint similarity loss arguments
                 fp_sim_weight=0.1,
                 fp_sim_loss_type='mse',
                 fp_sim_bin_size=0.05,
                 fp_sim_use_focal_loss=False,
                 fp_sim_focal_alpha=None,
                 fp_sim_focal_gamma=2.0,
                 fp_sim_use_focal_mse=False,
                 fp_sim_focal_mse_gamma=2.0,
                 # InfoNCE loss arguments
                 infonce_weight=0.1,
                 infonce_temperature=0.07):
        """
        Args:
            h_bin_size, c_bin_size: Bin sizes for H and C.
            h_max, c_max: Maximum ppm values for H and C.
            loss_type: 'hard_ce' for hard-label cross entropy, 'soft_ce' for
                Gaussian soft labels.
            sigma_h, sigma_c: Standard deviation of Gaussian soft labels.
            use_focal_loss: Whether to use focal loss for classification.
            focal_alpha, focal_gamma: Parameters for focal loss.
            cso_weight: Weight of the CSO loss.
            use_cso_loss: Whether to enable CSO loss.
            fp_sim_weight: Weight of the fingerprint similarity loss.
            fp_sim_loss_type: 'mse' or 'classification'.
            fp_sim_bin_size: Bin size for fingerprint similarity discretization.
            fp_sim_use_focal_loss: Whether to use focal loss for fingerprint
                similarity, only in classification mode.
            fp_sim_focal_alpha, fp_sim_focal_gamma: Parameters for fingerprint
                similarity focal loss.
            fp_sim_use_focal_mse: Whether to use focal MSE for fingerprint
                similarity, only in MSE mode.
            fp_sim_focal_mse_gamma: Gamma parameter for fingerprint similarity
                focal MSE.
            infonce_weight: Weight of the InfoNCE loss.
            infonce_temperature: Temperature parameter for InfoNCE.
        """
        super().__init__()

        # Initialize each loss component.
        self.classification_loss = ClassificationLoss(
            h_bin_size=h_bin_size,
            c_bin_size=c_bin_size,
            h_max=h_max,
            c_max=c_max,
            loss_type=loss_type,
            sigma_h=sigma_h,
            sigma_c=sigma_c,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )

        self.cso_loss = CSOLoss(
            weight=cso_weight,
            enabled=use_cso_loss
        )

        self.fingerprint_loss = FingerprintSimilarityLoss(
            weight=fp_sim_weight,
            loss_type=fp_sim_loss_type,
            bin_size=fp_sim_bin_size,
            use_focal_loss=fp_sim_use_focal_loss,
            focal_alpha=fp_sim_focal_alpha,
            focal_gamma=fp_sim_focal_gamma,
            use_focal_mse=fp_sim_use_focal_mse,
            focal_mse_gamma=fp_sim_focal_mse_gamma
        )

        self.infonce_loss = InfoNCELoss(
            weight=infonce_weight,
            temperature=infonce_temperature
        )

    def forward(self, h_logits, c_logits, sequence_output, shifts, labels, mask, types, model, padding_mask,
                fingerprints=None, h_c_shifts=None, c_shifts_list=None):
        """
        Args:
            h_logits: H classification logits [B, L, h_num_bins].
            c_logits: C classification logits [B, L, c_num_bins].
            sequence_output: Transformer encoder outputs [B, L, d_model].
            shifts: Raw input chemical shifts [B, L] in ppm.
            labels: Raw labels [B, L] in ppm.
            mask: Masked positions (bool).
            types: Atom types [B, L], where 0=H and 1=C.
            model: Model containing the cso_out layer for relative ordering.
            padding_mask: Padding mask [B, L].
            fingerprints: Morgan fingerprints [B, 2048], optional.
            h_c_shifts: C shift corresponding to each H peak [B, num_H], optional,
                used for InfoNCE.
            c_shifts_list: All C shifts for each sample as List[Tensor], optional,
                used for InfoNCE.

        Returns:
            total_loss: Total loss.
            loss_ce: Classification loss.
            loss_cso: CSO loss.
            loss_fp_sim: Fingerprint similarity loss.
            loss_infonce: InfoNCE loss.
        """
        device = h_logits.device

        # Check whether any positions are masked.
        num_masked = mask.sum()
        if num_masked == 0:
            return (torch.tensor(0.0, device=device), torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device), torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device))

        # Collect tensors at masked positions.
        masked_h_logits = h_logits[mask]  # [num_masked, h_num_bins]
        masked_c_logits = c_logits[mask]  # [num_masked, c_num_bins]
        masked_labels = labels[mask]      # [num_masked]
        masked_types = types[mask]        # [num_masked]

        # 1. Classification loss
        loss_ce = self.classification_loss(
            masked_h_logits, masked_c_logits,
            masked_labels, masked_types
        )

        # 2. CSO loss
        loss_cso = self.cso_loss(
            sequence_output, labels, mask, types, shifts, model
        )

        # 3. Fingerprint similarity loss
        loss_fp_sim = self.fingerprint_loss(
            sequence_output, padding_mask, fingerprints, model
        )

        # 4. InfoNCE loss
        loss_infonce = self.infonce_loss(
            sequence_output, types, mask, padding_mask,
            h_c_shifts, c_shifts_list, shifts
        )

        # Total loss with weights applied.
        total_loss = (loss_ce +
                      self.cso_loss.weight * loss_cso +
                      self.fingerprint_loss.weight * loss_fp_sim +
                      self.infonce_loss.weight * loss_infonce)

        # Return raw loss values (unweighted) for display and monitoring.
        return total_loss, loss_ce, loss_cso, loss_fp_sim, loss_infonce
