"""
Loss functions and evaluation metrics for functional group classification
Includes Focal Loss, LDAM, and optimal threshold finding from NMR2Struct
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, fbeta_score
from typing import Dict, List, Optional

# Import FocalLoss and LDAMLoss from local loss_fxns module
from loss_fxns import FocalLoss, LDAMLoss


class FGLoss(nn.Module):
    """
    Loss function for functional group classification with imbalance handling
    Enhanced with label smoothing for preventing overfitting
    """
    def __init__(
        self,
        strategy='weighted',  # 'none', 'weighted', 'focal', 'ldam'
        pos_weight: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_alpha_per_class: Optional[List[float]] = None,
        focal_gamma_per_class: Optional[List[float]] = None,
        cls_num_list: Optional[List] = None,
        ldam_max_m: float = 0.5,
        ldam_s: float = 30,
        use_logits: bool = True,
        label_smoothing: float = 0.0  # Label smoothing factor [0.0, 1.0]
    ):
        """
        Args:
            strategy: Loss strategy
            pos_weight: Positive class weights [num_classes]
            focal_alpha: Alpha for Focal Loss (single value for all classes)
            focal_gamma: Gamma for Focal Loss (single value for all classes)
            focal_alpha_per_class: Per-class alpha values [num_classes] (overrides focal_alpha)
            focal_gamma_per_class: Per-class gamma values [num_classes] (overrides focal_gamma)
            cls_num_list: Class counts for LDAM
            ldam_max_m: Max margin for LDAM
            ldam_s: Scale for LDAM
            use_logits: Whether model outputs logits
            label_smoothing: Label smoothing factor. Targets become:
                           - positive: 1.0 -> (1.0 - label_smoothing)
                           - negative: 0.0 -> label_smoothing
                           Typical values: 0.05 - 0.1
        """
        super().__init__()

        self.strategy = strategy
        self.use_logits = use_logits
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:
            print(f"✓ Label smoothing enabled: {label_smoothing}")
            print(f"  Positive labels: 1.0 -> {1.0 - label_smoothing}")
            print(f"  Negative labels: 0.0 -> {label_smoothing}")

        if strategy == 'focal':
            # Use per-class parameters if provided, otherwise use single values
            alpha = focal_alpha_per_class if focal_alpha_per_class is not None else focal_alpha
            gamma = focal_gamma_per_class if focal_gamma_per_class is not None else focal_gamma

            # Print configuration
            if focal_alpha_per_class is not None or focal_gamma_per_class is not None:
                print(f"Using Per-Class Focal Loss:")
                if focal_alpha_per_class is not None:
                    print(f"  alpha (per-class): {focal_alpha_per_class}")
                else:
                    print(f"  alpha (uniform): {focal_alpha}")
                if focal_gamma_per_class is not None:
                    print(f"  gamma (per-class): {focal_gamma_per_class}")
                else:
                    print(f"  gamma (uniform): {focal_gamma}")
                print(f"  use_logits: {use_logits}")
            else:
                print(f"Using Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}, use_logits={use_logits}")

            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma, use_logits=use_logits)

        elif strategy == 'ldam':
            if cls_num_list is None:
                raise ValueError("cls_num_list required for LDAM")
            print(f"Using LDAM Loss: max_m={ldam_max_m}, s={ldam_s}")
            self.loss_fn = LDAMLoss(cls_num_list, max_m=ldam_max_m, s=ldam_s)
            self.use_logits = True  # LDAM requires logits

        elif strategy == 'weighted':
            if pos_weight is None:
                raise ValueError("pos_weight required for weighted BCE")
            # Store pos_weight for potential filtering
            self.pos_weight = pos_weight
            if use_logits:
                print(f"Using BCEWithLogitsLoss with pos_weight")
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                print(f"Using weighted BCE")
                self.loss_fn = None

        elif strategy == 'none':
            if use_logits:
                print("Using BCEWithLogitsLoss (no weighting)")
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                print("Using BCE (no weighting)")
                self.loss_fn = nn.BCELoss()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, num_classes] logits or probabilities
            targets: [B, num_classes] binary labels (or -1 for excluded classes)

        Returns:
            loss: scalar
        """
        # Check if any classes are excluded (marked with -1)
        has_excluded = (targets == -1).any()

        if has_excluded:
            # Find valid classes (columns where all targets are not -1)
            # We check the first sample to identify excluded classes
            valid_class_mask = targets[0] != -1  # [num_classes]
            # Move mask to CPU for indexing parameters
            valid_class_mask_cpu = valid_class_mask.cpu()

            # Select only valid classes (keep [B, C_active] shape)
            predictions_valid = predictions[:, valid_class_mask]
            targets_valid = targets[:, valid_class_mask]

            # Apply label smoothing if enabled (only on valid targets)
            if self.label_smoothing > 0:
                targets_valid = targets_valid * (1.0 - self.label_smoothing) + self.label_smoothing

            # If using per-class parameters in focal loss, we need to filter them too
            if self.strategy == 'focal':
                # Create a new focal loss with filtered parameters
                from loss_fxns import FocalLoss

                # Get original alpha and gamma (use CPU mask for indexing)
                if hasattr(self.loss_fn, 'per_class_alpha') and self.loss_fn.per_class_alpha:
                    alpha_filtered = self.loss_fn.alpha[valid_class_mask_cpu]
                else:
                    alpha_filtered = self.loss_fn.alpha

                if hasattr(self.loss_fn, 'per_class_gamma') and self.loss_fn.per_class_gamma:
                    gamma_filtered = self.loss_fn.gamma[valid_class_mask_cpu]
                else:
                    gamma_filtered = self.loss_fn.gamma

                # Create temporary focal loss with filtered parameters
                temp_loss_fn = FocalLoss(
                    alpha=alpha_filtered,
                    gamma=gamma_filtered,
                    use_logits=self.loss_fn.use_logits
                )
                return temp_loss_fn(predictions_valid, targets_valid)

            elif self.strategy == 'weighted' and not self.use_logits and self.loss_fn is None:
                # Manual weighted BCE for probabilities
                pos_weight_filtered = self.pos_weight[valid_class_mask_cpu]
                bce = -(targets_valid * torch.log(predictions_valid + 1e-7) +
                        (1 - targets_valid) * torch.log(1 - predictions_valid + 1e-7))
                weight = targets_valid * pos_weight_filtered.unsqueeze(0) + (1 - targets_valid) * 1.0
                return (weight * bce).mean()

            elif self.strategy == 'weighted' and self.use_logits:
                # BCEWithLogitsLoss with filtered pos_weight
                pos_weight_filtered = self.loss_fn.pos_weight[valid_class_mask_cpu] if self.loss_fn.pos_weight is not None else None
                temp_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_filtered)
                return temp_loss_fn(predictions_valid, targets_valid)

            else:
                return self.loss_fn(predictions_valid, targets_valid)

        else:
            # No excluded classes, use original logic
            # Apply label smoothing if enabled
            if self.label_smoothing > 0:
                targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing

            if self.strategy == 'weighted' and not self.use_logits and self.loss_fn is None:
                # Manual weighted BCE for probabilities
                bce = -(targets * torch.log(predictions + 1e-7) +
                        (1 - targets) * torch.log(1 - predictions + 1e-7))
                weight = targets * self.pos_weight + (1 - targets) * 1.0
                return (weight * bce).mean()
            else:
                return self.loss_fn(predictions, targets)


def functional_group_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate metrics for functional group classification

    Args:
        predictions: [N, num_classes] probabilities
        targets: [N, num_classes] binary labels (or -1 for excluded classes)
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Filter out excluded classes (where targets == -1)
    valid_mask = targets != -1  # [N, num_classes]

    # Get indices of valid classes (classes that have at least one valid sample)
    valid_class_mask = valid_mask.any(axis=0)

    # Filter predictions and targets to include only valid classes
    predictions_filtered = predictions[:, valid_class_mask]
    targets_filtered = targets[:, valid_class_mask]

    # Apply threshold
    pred_binary = (predictions_filtered >= threshold).astype(int)
    target_binary = targets_filtered.astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(target_binary, pred_binary),
        'f1_micro': f1_score(target_binary, pred_binary, average='micro', zero_division=0),
        'f1_macro': f1_score(target_binary, pred_binary, average='macro', zero_division=0),
        'f1_samples': f1_score(target_binary, pred_binary, average='samples', zero_division=0),
        'f2_macro': fbeta_score(target_binary, pred_binary, beta=2.0, average='macro', zero_division=0),
        'recall_micro': recall_score(target_binary, pred_binary, average='micro', zero_division=0),
        'recall_macro': recall_score(target_binary, pred_binary, average='macro', zero_division=0),
        'precision_micro': precision_score(target_binary, pred_binary, average='micro', zero_division=0),
        'precision_macro': precision_score(target_binary, pred_binary, average='macro', zero_division=0),
    }

    return metrics


def per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    class_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-class metrics

    Args:
        predictions: [N, num_classes] probabilities
        targets: [N, num_classes] binary labels (or -1 for excluded classes)
        threshold: Classification threshold
        class_names: Optional class names

    Returns:
        Dictionary mapping class to metrics (excluded classes will be skipped)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    num_classes = targets.shape[1]
    if class_names is None:
        class_names = [f'FG_{i}' for i in range(num_classes)]

    per_class = {}

    for i, class_name in enumerate(class_names):
        # Check if this class is excluded (all targets are -1)
        if np.all(targets[:, i] == -1):
            continue  # Skip excluded classes

        y_true_raw = targets[:, i]
        y_pred_raw = predictions[:, i]

        # Filter out samples where target is -1
        valid_mask = y_true_raw != -1
        y_true = y_true_raw[valid_mask].astype(int)
        y_pred = (y_pred_raw[valid_mask] >= threshold).astype(int)

        if len(y_true) == 0:
            continue  # Skip if no valid samples

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[class_name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(y_true)),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }

    return per_class


def find_optimal_thresholds(
    all_probs: np.ndarray,
    all_targets: np.ndarray,
    num_thresholds: int = 100,
    metric: str = 'f1'
) -> Dict[int, float]:
    """
    Find optimal threshold for each class

    Args:
        all_probs: [N, num_classes] probabilities
        all_targets: [N, num_classes] binary labels
        num_thresholds: Number of threshold candidates
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Dictionary mapping class index to optimal threshold
    """

    num_classes = all_probs.shape[1]
    optimal_thresholds = {}
    threshold_candidates = np.linspace(0.1, 0.9, num_thresholds)

    print("\nFinding optimal thresholds...")
    print(f"Testing {num_thresholds} thresholds from 0.1 to 0.9")
    print(f"Optimizing for: {metric}")
    print("-" * 80)

    for class_idx in range(num_classes):
        class_probs = all_probs[:, class_idx]
        class_targets = all_targets[:, class_idx]

        # Filter out masked samples (where label is -1)
        valid_mask = class_targets >= 0
        class_probs = class_probs[valid_mask]
        class_targets = class_targets[valid_mask]

        # Skip if no valid samples or no positive samples
        if len(class_targets) == 0:
            optimal_thresholds[class_idx] = 0.5
            print(f"  Class {class_idx}: No valid samples (all masked), using 0.5")
            continue

        if class_targets.sum() == 0:
            optimal_thresholds[class_idx] = 0.5
            print(f"  Class {class_idx}: No positives, using 0.5")
            continue

        best_score = 0.0
        best_threshold = 0.5

        for threshold in threshold_candidates:
            predictions = (class_probs >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(class_targets, predictions, zero_division=0)
            elif metric == 'precision':
                score = precision_score(class_targets, predictions, zero_division=0)
            elif metric == 'recall':
                score = recall_score(class_targets, predictions, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        optimal_thresholds[class_idx] = best_threshold
        print(f"  Class {class_idx}: threshold={best_threshold:.3f}, {metric}={best_score:.4f}")

    return optimal_thresholds


def evaluate_with_optimal_thresholds(
    all_probs: np.ndarray,
    all_targets: np.ndarray,
    optimal_thresholds: Dict[int, float]
) -> Dict[str, float]:
    """
    Evaluate using class-specific optimal thresholds

    Args:
        all_probs: [N, num_classes] probabilities
        all_targets: [N, num_classes] binary labels
        optimal_thresholds: Dict mapping class to threshold

    Returns:
        Metrics dictionary
    """
    num_classes = all_probs.shape[1]
    predictions = np.zeros_like(all_probs)

    # Apply class-specific thresholds
    for class_idx in range(num_classes):
        threshold = optimal_thresholds.get(class_idx, 0.5)
        predictions[:, class_idx] = (all_probs[:, class_idx] >= threshold).astype(int)

    # Calculate metrics
    metrics = functional_group_metrics(
        torch.from_numpy(predictions),
        torch.from_numpy(all_targets)
    )

    return metrics


def evaluate_model(
    model,
    dataloader,
    device,
    use_optimal_thresholds=False,
    threshold=0.5,
    apply_sigmoid=True,
    label_key='fg_labels',
    metric = 'f1'
):
    """
    Evaluate model on dataloader

    Args:
        model: FunctionalGroupDecoder
        dataloader: DataLoader
        device: Device
        use_optimal_thresholds: Find optimal thresholds
        threshold: Fixed threshold (if not using optimal)
        apply_sigmoid: Apply sigmoid to outputs
        label_key: Key for labels in batch dict (default: 'fg_labels')

    Returns:
        overall_metrics, per_class_metrics
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            shifts = batch['shifts'].to(device)
            counts = batch['counts'].to(device)
            types = batch['types'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            fg_labels = batch[label_key].to(device)

            # Forward pass
            outputs = model(shifts, counts, types, padding_mask)

            # Apply sigmoid if needed
            if apply_sigmoid:
                outputs = torch.sigmoid(outputs)

            all_preds.append(outputs.cpu())
            all_targets.append(fg_labels.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Find optimal thresholds if requested
    if use_optimal_thresholds:
        print("\n" + "=" * 80)
        print("Finding optimal per-class thresholds")
        print("=" * 80)

        all_preds_np = all_preds.numpy()
        all_targets_np = all_targets.numpy()

        optimal_thresholds_dict = find_optimal_thresholds(
            all_preds_np, all_targets_np, 100,metric
        )

        # Evaluate with optimal thresholds
        print("\n" + "=" * 80)
        print("Evaluating with optimal thresholds")
        print("=" * 80)

        overall_metrics = evaluate_with_optimal_thresholds(
            all_preds_np, all_targets_np, optimal_thresholds_dict
        )

        # Calculate per-class metrics with optimal thresholds
        per_class = {}
        num_classes = all_preds.shape[1]

        for i in range(num_classes):
            threshold_i = optimal_thresholds_dict[i]
            class_preds = all_preds_np[:, i]
            class_targets = all_targets_np[:, i]

            pred_binary = (class_preds >= threshold_i).astype(int)

            tp = np.sum((class_targets == 1) & (pred_binary == 1))
            fp = np.sum((class_targets == 0) & (pred_binary == 1))
            tn = np.sum((class_targets == 0) & (pred_binary == 0))
            fn = np.sum((class_targets == 1) & (pred_binary == 0))

            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[f'FG_{i}'] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(np.sum(class_targets)),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'threshold': float(threshold_i)
            }
    else:
        # Use fixed threshold
        overall_metrics = functional_group_metrics(all_preds, all_targets, threshold=threshold)
        per_class = per_class_metrics(all_preds, all_targets, threshold=threshold)

    return overall_metrics, per_class


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Print metrics"""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")

    for key, value in sorted(metrics.items()):
        print(f"  {key:30s}: {value:6.4f}")

    print(f"{'=' * 60}\n")


def print_per_class_metrics(
    per_class: Dict[str, Dict],
    sort_by: str = 'f1',
    top_n: int = 10
):
    """Print per-class metrics"""
    print(f"\n{'=' * 100}")
    print(f"Per-Class Metrics (sorted by {sort_by}, showing top {top_n})")
    print(f"{'=' * 100}")

    # Sort
    sorted_classes = sorted(
        per_class.items(),
        key=lambda x: x[1].get(sort_by, 0),
        reverse=True
    )

    # Print header
    header = f"{'Class':15s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Supp':>7s} {'TP':>6s} {'FP':>6s} {'TN':>6s} {'FN':>6s}"
    if 'threshold' in next(iter(per_class.values())):
        header += f" {'Thr':>6s}"
    print(header)
    print("-" * 100)

    # Print top N
    for class_name, metrics in sorted_classes[:top_n]:
        line = (f"{class_name:15s} {metrics['accuracy']:7.4f} {metrics['precision']:7.4f} "
                f"{metrics['recall']:7.4f} {metrics['f1']:7.4f} {metrics['support']:7d} "
                f"{metrics['tp']:6d} {metrics['fp']:6d} {metrics['tn']:6d} {metrics['fn']:6d}")
        if 'threshold' in metrics:
            line += f" {metrics['threshold']:6.3f}"
        print(line)

    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    B, C = 32, 22
    logits = torch.randn(B, C)
    targets = torch.randint(0, 2, (B, C)).float()
    pos_weight = torch.ones(C) * 2.0

    # Test weighted BCE
    loss_fn = FGLoss(strategy='weighted', pos_weight=pos_weight, use_logits=True)
    loss = loss_fn(logits, targets)
    print(f"✓ Weighted BCE loss: {loss.item():.4f}")

    # Test Focal Loss
    probs = torch.sigmoid(logits)
    loss_fn = FGLoss(strategy='focal', focal_alpha=0.25, focal_gamma=2.0, use_logits=False)
    loss = loss_fn(probs, targets)
    print(f"✓ Focal loss: {loss.item():.4f}")

    # Test metrics
    metrics = functional_group_metrics(probs, targets, threshold=0.5)
    print(f"\n✓ Overall metrics computed: {len(metrics)} metrics")

    per_class = per_class_metrics(probs, targets, threshold=0.5)
    print(f"✓ Per-class metrics computed: {len(per_class)} classes")


def evaluate_model_gaussian(
    model,
    dataloader,
    device,
    use_optimal_thresholds=False,
    threshold=0.5,
    apply_sigmoid=True,
    label_key='fg_labels',
    metric='f1'
):
    """
    Evaluate Gaussian model on dataloader

    Args:
        model: FGClassifierGaussian
        dataloader: DataLoader with Gaussian embeddings
        device: Device
        use_optimal_thresholds: Find optimal thresholds
        threshold: Fixed threshold (if not using optimal)
        apply_sigmoid: Apply sigmoid to outputs
        label_key: Key for labels in batch dict (default: 'fg_labels')
        metric: Metric for optimal threshold finding ('f1', 'accuracy', etc.)

    Returns:
        overall_metrics, per_class_metrics
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            embedding = batch['embedding'].to(device)
            fg_labels = batch[label_key].to(device)

            # Forward pass
            outputs = model(embedding)

            # Apply sigmoid if needed
            if apply_sigmoid:
                outputs = torch.sigmoid(outputs)

            all_preds.append(outputs.cpu())
            all_targets.append(fg_labels.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Find optimal thresholds if requested
    if use_optimal_thresholds:
        print("\n" + "=" * 80)
        print("Finding optimal per-class thresholds")
        print("=" * 80)

        all_preds_np = all_preds.numpy()
        all_targets_np = all_targets.numpy()

        optimal_thresholds_dict = find_optimal_thresholds(
            all_preds_np, all_targets_np, 100, metric
        )

        # Evaluate with optimal thresholds
        print("\n" + "=" * 80)
        print("Evaluating with optimal thresholds")
        print("=" * 80)

        overall_metrics = evaluate_with_optimal_thresholds(
            all_preds_np, all_targets_np, optimal_thresholds_dict
        )

        # Calculate per-class metrics with optimal thresholds
        per_class = {}
        num_classes = all_preds.shape[1]

        for i in range(num_classes):
            threshold_i = optimal_thresholds_dict[i]
            class_preds = all_preds_np[:, i]
            class_targets = all_targets_np[:, i]

            pred_binary = (class_preds >= threshold_i).astype(int)

            tp = np.sum((class_targets == 1) & (pred_binary == 1))
            fp = np.sum((class_targets == 0) & (pred_binary == 1))
            tn = np.sum((class_targets == 0) & (pred_binary == 0))
            fn = np.sum((class_targets == 1) & (pred_binary == 0))

            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[f'FG_{i}'] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(np.sum(class_targets)),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'threshold': float(threshold_i)
            }
    else:
        # Use fixed threshold
        overall_metrics = functional_group_metrics(all_preds, all_targets, threshold=threshold)
        per_class = per_class_metrics(all_preds, all_targets, threshold=threshold)

    return overall_metrics, per_class
