import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Balancing factor for class weights. Can be a float or a list/tensor of per-class weights.
        gamma: Focusing parameter for modulating loss. Higher gamma increases focus on hard examples.
        reduction: 'mean', 'sum', or 'none'

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [N, C] Model output logits before softmax.
            targets: [N] Target class indices.

        Returns:
            loss: Scalar loss value.
        """
        # Compute cross-entropy loss without reduction.
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [N]

        # Compute probabilities.
        probs = F.softmax(logits, dim=1)  # [N, C]

        # Gather the probability p_t of the target class.
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]

        # Compute focal loss: FL = -(1 - p_t)^gamma * log(p_t).
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if provided.
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha contains per-class weights.
                if isinstance(self.alpha, list):
                    alpha_t = torch.tensor(self.alpha, device=logits.device)
                else:
                    alpha_t = self.alpha
                alpha_t = alpha_t.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply the requested reduction.
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
