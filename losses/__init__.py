"""
NMR Loss Functions Module

This module contains various loss functions for NMR foundation model training:
- FocalLoss: Focal loss for addressing class imbalance
- ClassificationLoss: Chemical shift classification loss
- CSOLoss: Chemical shift ordering loss
- FingerprintSimilarityLoss: Fingerprint similarity contrastive loss
- InfoNCELoss: H-C correspondence learning loss
- NMRLoss: Combined loss function
"""

from losses.focal_loss import FocalLoss
from losses.classification_loss import ClassificationLoss
from losses.cso_loss import CSOLoss
from losses.fingerprint_loss import FingerprintSimilarityLoss
from losses.infonce_loss import InfoNCELoss
from losses.nmr_loss import NMRLoss

__all__ = [
    'FocalLoss',
    'ClassificationLoss',
    'CSOLoss',
    'FingerprintSimilarityLoss',
    'InfoNCELoss',
    'NMRLoss',
]
