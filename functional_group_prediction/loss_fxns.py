import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import numpy as np

class SubsWeightedBCELoss(nn.Module):
    def __init__(self,
                 weights: Optional[Tensor] = None):
        super().__init__()
        self.weights = weights

    def forward(self,
                y_pred: Tensor,
                y_true: Tensor) -> Tensor:
        '''
        Computes the weighted BCE Loss with a different 0/1 class weight per substructure.

        y_pred: (batch_size, num_substructures), the predicted probabilities for each substructure
        y_true: (batch_size, num_substructures), the true substructures present in each molecule
        weights: (batch_size, num_substructures, 2), the weights for each substructure. Each row i represents the
            0 and 1 weight for substructure i. If weights are not given, then the loss is unweighted.
        reduction: The method for reducing the loss. Defaults to 'mean', but can be 'mean' or 'sum'.
        '''
        criterion = nn.BCELoss(reduction = 'none')
        unweighted_loss = criterion(y_pred, y_true)
        #Compute the weight for each substructure now
        if self.weights is not None:
            w = y_true * self.weights[:,:, 1] + (1 - y_true) * self.weights[:,:, 0]
            weighted_loss = w * unweighted_loss
        else:
            weighted_loss = 1 * unweighted_loss
        #Averaged across substructures
        rowwise_meaned = torch.mean(weighted_loss, dim = 0)
        assert(rowwise_meaned.shape == (957,))
        #Sum for total loss
        tot_loss = torch.sum(rowwise_meaned)
        return tot_loss

CrossEntropyLoss = nn.CrossEntropyLoss
BCELoss = nn.BCELoss

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               Can be a scalar (same for all classes) or tensor of shape [num_classes]
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
               Can be a scalar (same for all classes) or tensor of shape [num_classes]
        reduction: 'none' | 'mean' | 'sum'
        use_logits: If True, inputs are logits; if False, inputs are probabilities
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', use_logits=False):
        super().__init__()

        # Handle per-class alpha
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
            self.per_class_alpha = True
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
            self.per_class_alpha = True
        else:
            self.alpha = float(alpha)
            self.per_class_alpha = False

        # Handle per-class gamma
        if isinstance(gamma, (list, tuple)):
            self.gamma = torch.tensor(gamma, dtype=torch.float32)
            self.per_class_gamma = True
        elif isinstance(gamma, torch.Tensor):
            self.gamma = gamma.float()
            self.per_class_gamma = True
        else:
            self.gamma = float(gamma)
            self.per_class_gamma = False

        self.reduction = reduction
        self.use_logits = use_logits

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - predicted logits or probabilities
            targets: (batch_size, num_classes) - ground truth labels
        """
        # Move alpha and gamma to same device as inputs if needed
        if self.per_class_alpha and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        if self.per_class_gamma and self.gamma.device != inputs.device:
            self.gamma = self.gamma.to(inputs.device)

        if self.use_logits:
            # Compute BCE loss from logits
            bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            # Convert logits to probabilities for focal term
            p = torch.sigmoid(inputs)
        else:
            # Compute BCE loss from probabilities
            bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
            p = inputs

        # Clamp probabilities for numerical stability
        p = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)

        # Compute p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # Clamp p_t to ensure (1 - p_t) is non-negative for power operation
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)

        # Compute focal term: (1 - p_t)^gamma
        if self.per_class_gamma:
            # Broadcast gamma to shape [1, num_classes] for per-class gamma
            gamma_broadcast = self.gamma.view(1, -1)
            focal_term = (1 - p_t) ** gamma_broadcast
        else:
            focal_term = (1 - p_t) ** self.gamma

        # Compute alpha term
        if self.per_class_alpha:
            # Broadcast alpha to shape [1, num_classes] for per-class alpha
            alpha_broadcast = self.alpha.view(1, -1)
            alpha_t = alpha_broadcast * targets + (1 - alpha_broadcast) * (1 - targets)
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_t * focal_term * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LDAMLoss(nn.Module):
    """
    Label Distribution Aware Margin (LDAM) Loss

    Applies different margins to different classes based on their frequency.
    Rare classes get larger margins to make them easier to classify.

    Reference: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (NeurIPS 2019)
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        """
        Args:
            cls_num_list: List of sample counts for each class
            max_m: Maximum margin
            s: Scale parameter
        """
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits (before sigmoid)
            targets: (batch_size, num_classes) - ground truth labels
        """
        # Move m_list to same device as inputs
        if self.m_list.device != inputs.device:
            self.m_list = self.m_list.to(inputs.device)

        # Apply margin
        batch_m = torch.matmul(targets, self.m_list.view(-1, 1)).view(-1, 1)
        inputs_m = inputs - batch_m

        # For multi-label, we apply margin to positive samples only
        output = torch.where(targets > 0, inputs_m, inputs)

        # BCEWithLogitsLoss
        loss = nn.functional.binary_cross_entropy_with_logits(output / self.s, targets)

        return loss


class MultiTaskLoss(nn.Module):
    def __init__(self,
                 ignore_index: int,
                 substructure_weight: float,
                 structure_weight: float,
                 pos_weight: Optional[Tensor] = None,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 use_ldam: bool = False,
                 cls_num_list: Optional[list] = None,
                 ldam_max_m: float = 0.5,
                 ldam_s: float = 30,
                 use_logits: bool = False):
        """
        Multi-task loss with support for class imbalance handling

        Args:
            ignore_index: Index to ignore in structure loss
            substructure_weight: Weight for substructure loss
            structure_weight: Weight for structure loss
            pos_weight: Weight for positive class in BCE loss (shape: [num_fg])
            use_focal_loss: Whether to use Focal Loss instead of BCE
            focal_alpha: Alpha parameter for Focal Loss
            focal_gamma: Gamma parameter for Focal Loss
            use_ldam: Whether to use LDAM loss
            cls_num_list: List of positive sample counts for each class (for LDAM)
            ldam_max_m: Maximum margin for LDAM
            ldam_s: Scale parameter for LDAM
            use_logits: Whether model outputs logits (True) or probabilities (False)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.SUBSTRUCTURE_weight = substructure_weight
        self.STRUCT_weight = structure_weight
        self.structure_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.use_logits = use_logits

        # Choose loss function for substructure prediction
        self.use_focal_loss = use_focal_loss
        self.use_ldam = use_ldam

        if use_ldam:
            if cls_num_list is None:
                raise ValueError("cls_num_list must be provided when use_ldam=True")
            print(f"Using LDAM Loss with max_m={ldam_max_m}, s={ldam_s}")
            self.substructure_loss = LDAMLoss(cls_num_list, max_m=ldam_max_m, s=ldam_s)
            self.use_logits = True  # LDAM requires logits
        elif use_focal_loss:
            print(f"Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}, use_logits={use_logits}")
            self.substructure_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, use_logits=use_logits)
        elif pos_weight is not None:
            if use_logits:
                print(f"Using BCEWithLogitsLoss with pos_weight (shape: {pos_weight.shape})")
                self.substructure_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                print(f"Using weighted BCELoss with pos_weight (shape: {pos_weight.shape})")
                self.pos_weight = pos_weight
                self.substructure_loss = None  # We'll compute weighted BCE manually
        else:
            if use_logits:
                print("Using BCEWithLogitsLoss (no class weighting)")
                self.substructure_loss = nn.BCEWithLogitsLoss()
            else:
                print("Using standard BCELoss (no class weighting)")
                self.substructure_loss = nn.BCELoss()
            self.pos_weight = None

    def forward(self,
                mode: str,
                y_pred: Tensor,
                y_true: Tensor) -> Tensor:
        if mode == 'substructure':
            if self.use_ldam or (self.use_logits and self.substructure_loss is not None):
                # Direct loss computation (for LDAM, BCEWithLogitsLoss, etc.)
                return self.SUBSTRUCTURE_weight * self.substructure_loss(y_pred, y_true)
            elif self.use_focal_loss:
                return self.SUBSTRUCTURE_weight * self.substructure_loss(y_pred, y_true)
            elif self.pos_weight is not None and self.substructure_loss is None:
                # Manual weighted BCE (for when using probabilities with pos_weight)
                bce = -(y_true * torch.log(y_pred + 1e-7) + (1 - y_true) * torch.log(1 - y_pred + 1e-7))
                weight = y_true * self.pos_weight + (1 - y_true) * 1.0
                weighted_bce = weight * bce
                return self.SUBSTRUCTURE_weight * weighted_bce.mean()
            else:
                return self.SUBSTRUCTURE_weight * self.substructure_loss(y_pred, y_true)
        elif mode == 'structure':
            return self.STRUCT_weight * self.structure_loss(y_pred, y_true)
