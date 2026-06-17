"""
Inference script for testing Functional Group Decoder on test dataset
Loads best model checkpoint and evaluates on test set with extended metrics
Uses optimal thresholds from validation set
"""
import torch
import yaml
import json
import os
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
FALLBACK_FUNCG_DIR = REPO_ROOT.parent / "NMR-foundation" / "functional_group_code"

for path in (CURRENT_DIR, REPO_ROOT, FALLBACK_FUNCG_DIR):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.append(path_str)

from fg_decoder import create_fg_decoder
from fg_dataloader import FGDataset
from fg_loss_metrics import (
    functional_group_metrics,
    per_class_metrics,
    find_optimal_thresholds,
    evaluate_with_optimal_thresholds,
    print_metrics,
    print_per_class_metrics
)


def resolve_local_path(path_value, base_dir=CURRENT_DIR):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def select_device():
    """Select a valid runtime device."""
    if not torch.cuda.is_available():
        return torch.device('cpu')

    requested_index = os.environ.get('ULTRANMR_CUDA_DEVICE', '0')
    try:
        requested_index = int(requested_index)
    except ValueError:
        requested_index = 0

    device_count = torch.cuda.device_count()
    if 0 <= requested_index < device_count:
        return torch.device(f'cuda:{requested_index}')

    print(
        f"⚠ Requested CUDA device {requested_index} is unavailable "
        f"(found {device_count} device(s)). Falling back to cuda:0."
    )
    return torch.device('cuda:0')


def calculate_macro_micro_accuracy(predictions, targets, threshold=0.5, exclude_classes=None):
    """
    Calculate macro and micro accuracy

    Args:
        predictions: [N, num_classes] probabilities
        targets: [N, num_classes] binary labels
        threshold: Classification threshold or dict mapping class to threshold
        exclude_classes: List of class indices to exclude from calculations

    Returns:
        dict with macro_acc and micro_acc
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    num_classes = targets.shape[1]
    if exclude_classes is None:
        exclude_classes = []

    # Apply threshold(s)
    if isinstance(threshold, dict):
        # Per-class thresholds
        pred_binary = np.zeros_like(predictions)
        for class_idx in range(num_classes):
            thresh = threshold.get(class_idx, 0.5)
            pred_binary[:, class_idx] = (predictions[:, class_idx] >= thresh).astype(int)
    else:
        # Uniform threshold
        pred_binary = (predictions >= threshold).astype(int)

    target_binary = targets.astype(int)

    # Create mask for classes to include
    class_mask = np.ones(num_classes, dtype=bool)
    for exc_class in exclude_classes:
        class_mask[exc_class] = False

    # Micro accuracy: overall accuracy across all samples and classes (excluding specified classes)
    # (TP + TN) / (TP + TN + FP + FN) for included classes only
    included_pred = pred_binary[:, class_mask]
    included_target = target_binary[:, class_mask]
    micro_acc = np.mean(included_pred == included_target)

    # Macro accuracy: average of per-class accuracies (excluding specified classes)
    class_accuracies = []
    for i in range(num_classes):
        if i in exclude_classes:
            continue
        y_true = target_binary[:, i]
        y_pred = pred_binary[:, i]

        # Per-class accuracy
        class_acc = np.mean(y_true == y_pred)
        class_accuracies.append(class_acc)

    macro_acc = np.mean(class_accuracies)

    return {
        'micro_acc': float(micro_acc),
        'macro_acc': float(macro_acc)
    }


def inference_test_dataset(checkpoint_path, config_path, exclude_classes=[15, 16]):
    """
    Run inference on test dataset

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        exclude_classes: List of class indices to exclude from metric calculations
    """
    print("\n" + "=" * 80)
    print("Functional Group Decoder - Test Set Inference")
    print("=" * 80)

    if exclude_classes:
        print(f"⚠ Excluding classes {exclude_classes} from metric calculations")

    config_path = resolve_local_path(config_path)
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config_path.parent / checkpoint_path).resolve()

    # Load config
    print(f"\nLoading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for section, key in [
        ('data', 'test_jsonl'),
        ('model', 'pretrained_checkpoint'),
        ('training', 'save_dir'),
    ]:
        value = config.get(section, {}).get(key)
        if isinstance(value, str) and not os.path.isabs(value):
            config[section][key] = str((config_path.parent / value).resolve())

    # Setup device
    device = select_device()
    print(f"Device: {device}")

    # Create test dataloader
    print("\n" + "=" * 80)
    print("Creating test dataloader...")
    print("=" * 80)

    test_dataset = FGDataset(
        jsonl_path=config['data']['test_jsonl'],
        max_peaks=config['data'].get('max_peaks', 200),
        transform=None,
        exclude_classes=exclude_classes
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print(f"✓ Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "=" * 80)
    print("Creating model...")
    print("=" * 80)

    model = create_fg_decoder(
        checkpoint_path=config['model']['pretrained_checkpoint'],
        config=config['model']['encoder_config'],
        num_functional_groups=config['model']['num_functional_groups'],
        freeze_encoder=config['model'].get('freeze_encoder', False),
        use_logits=config['training']['loss'].get('use_logits', True),
        device=device
    )

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        print(f"  Checkpoint val F1 macro: {checkpoint['val_metrics'].get('f1_macro', 'N/A'):.4f}")

    # Run inference
    print("\n" + "=" * 80)
    print("Running inference on test set...")
    print("=" * 80)

    all_preds = []
    all_targets = []

    apply_sigmoid = config['training']['loss'].get('use_logits', True)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            shifts = batch['shifts'].to(device)
            counts = batch['counts'].to(device)
            types = batch['types'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            fg_labels = batch['fg_labels'].to(device)

            # Forward pass
            outputs = model(shifts, counts, types, padding_mask)

            # Apply sigmoid if needed
            if apply_sigmoid:
                outputs = torch.sigmoid(outputs)

            all_preds.append(outputs.cpu())
            all_targets.append(fg_labels.cpu())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    print(f"✓ Inference complete")
    print(f"  Predictions shape: {all_preds.shape}")
    print(f"  Targets shape: {all_targets.shape}")

    # Convert to numpy for threshold finding
    all_preds_np = all_preds.numpy()
    all_targets_np = all_targets.numpy()

    # Exclude specified classes by setting them to -1 in targets
    if exclude_classes:
        for exc_class in exclude_classes:
            all_targets_np[:, exc_class] = -1
        print(f"✓ Excluded classes {exclude_classes} from metric calculations (set to -1)")

    # Evaluate with fixed threshold
    print("\n" + "=" * 80)
    print("Evaluation with fixed threshold (0.5)")
    print("=" * 80)

    fixed_threshold = config['evaluation'].get('threshold', 0.5)
    fixed_metrics = functional_group_metrics(all_preds, all_targets, threshold=fixed_threshold)
    fixed_acc_metrics = calculate_macro_micro_accuracy(all_preds, all_targets, threshold=fixed_threshold)

    # Combine metrics
    fixed_metrics.update(fixed_acc_metrics)

    print_metrics(fixed_metrics, title=f"Test Metrics (Fixed Threshold = {fixed_threshold})")

    # Per-class metrics with fixed threshold
    fixed_per_class = per_class_metrics(all_preds, all_targets, threshold=fixed_threshold)
    print_per_class_metrics(fixed_per_class, sort_by='f1', top_n=22)

    # Save results with fixed threshold
    save_dir = os.path.dirname(str(checkpoint_path))
    fixed_results_path = os.path.join(save_dir, 'test_inference_fixed_threshold.json')
    fixed_results = {
        'threshold': fixed_threshold,
        'overall_metrics': fixed_metrics,
        'per_class_metrics': fixed_per_class
    }

    with open(fixed_results_path, 'w') as f:
        json.dump(fixed_results, f, indent=2)
    print(f"\n✓ Fixed threshold results saved: {fixed_results_path}")

    # Evaluate with optimal thresholds from validation set
    use_optimal_thresholds = config['evaluation'].get('use_optimal_thresholds', True)

    if use_optimal_thresholds:
        print("\n" + "=" * 80)
        print("Loading optimal per-class thresholds from validation set")
        print("=" * 80)

        # Load thresholds from validation metrics file
        val_metrics_path = os.path.join(save_dir, f"val_metrics_epoch_{checkpoint['epoch']}.json")

        if not os.path.exists(val_metrics_path):
            print(f"⚠ Warning: Validation metrics file not found: {val_metrics_path}")
            print("  Falling back to searching thresholds on test set (NOT RECOMMENDED)")
            optimal_thresholds_dict = find_optimal_thresholds(
                all_preds_np, all_targets_np, num_thresholds=100, metric='f1'
            )
        else:
            print(f"✓ Loading thresholds from: {val_metrics_path}")
            with open(val_metrics_path, 'r') as f:
                val_metrics = json.load(f)

            # Extract per-class thresholds
            optimal_thresholds_dict = {}
            num_classes = all_preds.shape[1]

            for i in range(num_classes):
                fg_key = f'FG_{i}'
                if fg_key in val_metrics['per_class']:
                    optimal_thresholds_dict[i] = val_metrics['per_class'][fg_key]['threshold']
                else:
                    print(f"  ⚠ Warning: Threshold for {fg_key} not found, using default 0.5")
                    optimal_thresholds_dict[i] = 0.5

            print(f"✓ Loaded {len(optimal_thresholds_dict)} class thresholds from validation set")
            # Print some threshold examples
            print(f"  Example thresholds: FG_0={optimal_thresholds_dict[0]:.3f}, "
                  f"FG_1={optimal_thresholds_dict[1]:.3f}, FG_2={optimal_thresholds_dict[2]:.3f}")

        # Evaluate with optimal thresholds
        print("\n" + "=" * 80)
        print("Evaluation with optimal thresholds")
        print("=" * 80)

        optimal_metrics = evaluate_with_optimal_thresholds(
            all_preds_np, all_targets_np, optimal_thresholds_dict
        )

        # Calculate macro/micro accuracy with optimal thresholds
        optimal_acc_metrics = calculate_macro_micro_accuracy(
            all_preds_np, all_targets_np, threshold=optimal_thresholds_dict
        )

        # Combine metrics
        optimal_metrics.update(optimal_acc_metrics)

        print_metrics(optimal_metrics, title="Test Metrics (Optimal Thresholds)")

        # Calculate per-class metrics with optimal thresholds
        optimal_per_class = {}
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

            optimal_per_class[f'FG_{i}'] = {
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

        print_per_class_metrics(optimal_per_class, sort_by='f1', top_n=22)

        # Save results with optimal thresholds
        optimal_results_path = os.path.join(save_dir, 'test_inference_optimal_thresholds.json')
        optimal_results = {
            'optimal_thresholds': {int(k): float(v) for k, v in optimal_thresholds_dict.items()},
            'overall_metrics': optimal_metrics,
            'per_class_metrics': optimal_per_class
        }

        with open(optimal_results_path, 'w') as f:
            json.dump(optimal_results, f, indent=2)
        print(f"\n✓ Optimal threshold results saved: {optimal_results_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Model: {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"\nFixed Threshold ({fixed_threshold}):")
    print(f"  F1 Macro:    {fixed_metrics['f1_macro']:.4f}")
    print(f"  F1 Micro:    {fixed_metrics['f1_micro']:.4f}")
    print(f"  Macro Acc:   {fixed_metrics['macro_acc']:.4f}")
    print(f"  Micro Acc:   {fixed_metrics['micro_acc']:.4f}")
    print(f"  Precision M: {fixed_metrics['precision_macro']:.4f}")
    print(f"  Recall M:    {fixed_metrics['recall_macro']:.4f}")

    if use_optimal_thresholds:
        print(f"\nOptimal Thresholds:")
        print(f"  F1 Macro:    {optimal_metrics['f1_macro']:.4f}")
        print(f"  F1 Micro:    {optimal_metrics['f1_micro']:.4f}")
        print(f"  Macro Acc:   {optimal_metrics['macro_acc']:.4f}")
        print(f"  Micro Acc:   {optimal_metrics['micro_acc']:.4f}")
        print(f"  Precision M: {optimal_metrics['precision_macro']:.4f}")
        print(f"  Recall M:    {optimal_metrics['recall_macro']:.4f}")

    print("\n" + "=" * 80)
    print("Inference completed!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on test dataset')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='../model_checkpoint/checkpoints_fg/fg_best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    inference_test_dataset(args.checkpoint, args.config)
