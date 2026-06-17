"""
Data loader for functional group classification
Load balanced JSONL data and convert to UltraNMR format
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

class FGDataset(Dataset):
    """
    Functional Group Classification Dataset
    Load from JSONL files and convert to UltraNMR format
    """
    def __init__(
        self,
        jsonl_path: str,
        max_peaks: int = 200,
        pad_value: float = 0.0,
        transform=None,
        exclude_classes: Optional[List[int]] = None
    ):
        """
        Args:
            jsonl_path: Path to JSONL file
            max_peaks: Maximum number of peaks (for padding)
            pad_value: Padding value
            transform: Optional data augmentation
            exclude_classes: List of class indices to exclude from training/evaluation (e.g., [15, 16])
        """
        super().__init__()

        print(f"\nLoading data from: {jsonl_path}")

        data_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {jsonl_path}:{line_no}") from exc

                required_keys = {'smiles', 'h_shift', 'c_shift', 'fg_onehot'}
                missing_keys = required_keys - item.keys()
                if missing_keys:
                    missing_str = ', '.join(sorted(missing_keys))
                    raise ValueError(f"Missing keys [{missing_str}] at {jsonl_path}:{line_no}")

                data_list.append(item)

        # Extract data from list of dicts
        self.smiles_list = [item['smiles'] for item in data_list]
        self.h_shifts_list = [item['h_shift'] for item in data_list]
        self.c_shifts_list = [item['c_shift'] for item in data_list]
        self.functional_groups = np.array([item['fg_onehot'] for item in data_list], dtype=np.float32)

        self.max_peaks = max_peaks
        self.pad_value = pad_value
        self.transform = transform
        self.exclude_classes = exclude_classes if exclude_classes is not None else []

        # Mask excluded classes (set to -1 to ignore in loss computation)
        if self.exclude_classes:
            print(f"\n⚠ Excluding classes: {self.exclude_classes}")
            for cls_idx in self.exclude_classes:
                self.functional_groups[:, cls_idx] = -1
            print(f"  - These classes will be ignored in training and evaluation")

        print(f"✓ Loaded {len(self.smiles_list)} samples")
        print(f"  - Functional groups shape: {self.functional_groups.shape}")
        print(f"  - Num FG classes: {self.functional_groups.shape[1]}")
        print(f"  - Active classes: {self.functional_groups.shape[1] - len(self.exclude_classes)}")

        # Compute class statistics (excluding masked classes)
        fg_counts = self.functional_groups.copy()
        # Only count non-masked classes (where label is 0 or 1, not -1)
        for i in range(fg_counts.shape[1]):
            valid_mask = fg_counts[:, i] >= 0
            fg_counts[valid_mask, i] = self.functional_groups[valid_mask, i]
            if i in self.exclude_classes:
                fg_counts[:, i] = 0
        fg_counts = fg_counts.sum(axis=0)

        print(f"\nFunctional Group Statistics (excluding masked classes):")
        # Get statistics only for active classes
        active_counts = [fg_counts[i] for i in range(len(fg_counts)) if i not in self.exclude_classes]
        if active_counts:
            print(f"  - Min count: {min(active_counts):.0f}")
            print(f"  - Max count: {max(active_counts):.0f}")
            print(f"  - Mean count: {np.mean(active_counts):.0f}")

        # Show excluded classes with zero counts
        if self.exclude_classes:
            print(f"\nExcluded classes (masked):")
            for cls_idx in self.exclude_classes:
                print(f"  - Class {cls_idx}: masked (originally {(np.array([item['fg_onehot'] for item in data_list])[:, cls_idx]).sum():.0f} samples)")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        Returns:
            shifts: [L] Combined H and C shifts
            counts: [L] All ones (no count info in NMR2Struct data)
            types: [L] Atom types (0=H, 1=C)
            padding_mask: [L] Padding mask
            functional_groups: [num_fg] Binary labels
        """
        h_shifts = self.h_shifts_list[idx]
        c_shifts = self.c_shifts_list[idx]
        fg_labels = self.functional_groups[idx]

        # Convert to numpy arrays if not already
        if isinstance(h_shifts, list):
            h_shifts = np.array(h_shifts, dtype=np.float32)
        if isinstance(c_shifts, list):
            c_shifts = np.array(c_shifts, dtype=np.float32)
        if isinstance(fg_labels, list):
            fg_labels = np.array(fg_labels, dtype=np.float32)

        # Combine H and C shifts
        num_h = len(h_shifts)
        num_c = len(c_shifts)
        total_peaks = num_h + num_c

        # Create combined arrays
        shifts = np.zeros(self.max_peaks, dtype=np.float32)
        types = np.zeros(self.max_peaks, dtype=np.int64)
        counts = np.ones(self.max_peaks, dtype=np.float32)
        padding_mask = np.ones(self.max_peaks, dtype=bool)

        # Fill in H shifts (type=0)
        if num_h > 0:
            end_h = min(num_h, self.max_peaks)
            shifts[:end_h] = h_shifts[:end_h]
            types[:end_h] = 0  # H type
            padding_mask[:end_h] = False

        # Fill in C shifts (type=1)
        if num_c > 0 and num_h < self.max_peaks:
            start_c = num_h
            end_c = min(num_h + num_c, self.max_peaks)
            actual_c = end_c - start_c
            shifts[start_c:end_c] = c_shifts[:actual_c]
            types[start_c:end_c] = 1  # C type
            padding_mask[start_c:end_c] = False

        # Apply augmentation if provided
        if self.transform is not None:
            shifts, counts, types = self.transform(shifts, counts, types, padding_mask)

        # Convert to tensors
        shifts = torch.from_numpy(shifts)
        counts = torch.from_numpy(counts)
        types = torch.from_numpy(types)
        padding_mask = torch.from_numpy(padding_mask)
        fg_labels = torch.from_numpy(fg_labels).float()

        return {
            'shifts': shifts,
            'counts': counts,
            'types': types,
            'padding_mask': padding_mask,
            'fg_labels': fg_labels
        }


class GaussianNoise:
    """Add Gaussian noise to chemical shifts for data augmentation"""
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, shifts, counts, types, padding_mask):
        # Only add noise to non-padding positions
        noise = np.random.normal(0, self.std, shifts.shape).astype(np.float32)
        noise[padding_mask] = 0
        shifts = shifts + noise
        return shifts, counts, types


class RandomScale:
    """Randomly scale peak intensities"""
    def __init__(self, scale_range: Tuple[float, float] = (0.95, 1.05)):
        self.scale_range = scale_range

    def __call__(self, shifts, counts, types, padding_mask):
        scale = np.random.uniform(*self.scale_range)
        counts = counts * scale
        return shifts, counts, types


class ComposeTransforms:
    """Compose multiple transforms"""
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, shifts, counts, types, padding_mask):
        for transform in self.transforms:
            shifts, counts, types = transform(shifts, counts, types, padding_mask)
        return shifts, counts, types


def create_dataloaders(
    train_jsonl: str,
    val_jsonl: str,
    test_jsonl: str,
    batch_size: int = 64,
    num_workers: int = 4,
    max_peaks: int = 200,
    augment_train: bool = True,
    noise_std: float = 0.02,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    exclude_classes: Optional[List[int]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Args:
        train_jsonl: Path to training JSONL file
        val_jsonl: Path to validation JSONL file
        test_jsonl: Path to test JSONL file
        batch_size: Batch size
        num_workers: Number of workers for dataloader
        max_peaks: Maximum number of peaks
        augment_train: Apply augmentation to training data
        noise_std: Standard deviation for Gaussian noise
        scale_range: Range for random scaling
        exclude_classes: List of class indices to exclude from training/evaluation (e.g., [15, 16])

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create transforms for training
    train_transform = None
    if augment_train:
        train_transform = ComposeTransforms([
            GaussianNoise(std=noise_std),
            RandomScale(scale_range=scale_range)
        ])
        print(f"\n✓ Training augmentation enabled:")
        print(f"  - Gaussian noise: std={noise_std}")
        print(f"  - Random scaling: range={scale_range}")

    # Create datasets
    train_dataset = FGDataset(
        jsonl_path=train_jsonl,
        max_peaks=max_peaks,
        transform=train_transform,
        exclude_classes=exclude_classes
    )
    
    # subset_size = int(0.4 * len(train_dataset))
    # indices = random.sample(range(len(train_dataset)), subset_size)
    # train_dataset = Subset(train_dataset, indices=indices)

    val_dataset = FGDataset(
        jsonl_path=val_jsonl,
        max_peaks=max_peaks,
        transform=None,
        exclude_classes=exclude_classes
    )

    test_dataset = FGDataset(
        jsonl_path=test_jsonl,
        max_peaks=max_peaks,
        transform=None,
        exclude_classes=exclude_classes
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n{'=' * 60}")
    print(f"Dataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"{'=' * 60}\n")

    return train_loader, val_loader, test_loader


def compute_class_weights(dataloader, num_classes: int, device='cpu', exclude_classes: Optional[List[int]] = None):
    """
    Compute class weights for handling imbalance

    Args:
        dataloader: DataLoader
        num_classes: Number of classes
        device: Device to put weights on
        exclude_classes: List of class indices to exclude (will be set to 1.0)

    Returns:
        pos_weight: Tensor of shape (num_classes,)
        pos_counts: List of positive counts
    """
    print("\nComputing class weights...")

    if exclude_classes is None:
        exclude_classes = []

    pos_counts = torch.zeros(num_classes)
    neg_counts = torch.zeros(num_classes)

    for batch in dataloader:
        fg_labels = batch['fg_labels']  # [B, num_classes]
        # Only count valid labels (not -1 for excluded classes)
        for i in range(num_classes):
            if i not in exclude_classes:
                valid_mask = fg_labels[:, i] >= 0
                pos_counts[i] += fg_labels[valid_mask, i].sum().cpu()
                neg_counts[i] += (1 - fg_labels[valid_mask, i]).sum().cpu()

    # Compute pos_weight = neg_count / pos_count
    pos_weight = neg_counts / (pos_counts + 1e-6)

    # Set weight to 1.0 for excluded classes (will be ignored anyway)
    for cls_idx in exclude_classes:
        pos_weight[cls_idx] = 1.0

    # Clip extreme weights
    pos_weight = torch.clamp(pos_weight, min=1.0, max=100.0)

    # Get statistics only for active classes
    active_indices = [i for i in range(num_classes) if i not in exclude_classes]
    if active_indices:
        active_pos_counts = pos_counts[active_indices]
        active_pos_weight = pos_weight[active_indices]

        print(f"\nClass Distribution (excluding {len(exclude_classes)} classes):")
        print(f"  Active classes: {len(active_indices)}")
        print(f"  Min pos_count: {active_pos_counts.min().item():.0f}")
        print(f"  Max pos_count: {active_pos_counts.max().item():.0f}")
        print(f"  Min pos_weight: {active_pos_weight.min().item():.2f}")
        print(f"  Max pos_weight: {active_pos_weight.max().item():.2f}")
        print(f"  Mean pos_weight: {active_pos_weight.mean().item():.2f}")

        # Print most imbalanced classes
        print(f"\nMost imbalanced classes (top 5):")
        sorted_indices = torch.argsort(pos_weight, descending=True)
        count = 0
        for idx in sorted_indices:
            idx = idx.item()
            if idx not in exclude_classes:
                print(f"  FG_{idx}: pos={pos_counts[idx].item():.0f}, "
                      f"neg={neg_counts[idx].item():.0f}, weight={pos_weight[idx].item():.2f}")
                count += 1
                if count >= 5:
                    break

    if exclude_classes:
        print(f"\nExcluded classes (weight set to 1.0):")
        for cls_idx in exclude_classes:
            print(f"  FG_{cls_idx}: excluded")

    return pos_weight.to(device), pos_counts.numpy().tolist()


if __name__ == "__main__":
    # Test dataloader
    train_jsonl = str((CURRENT_DIR.parent / "data" / "NMRGym_train_balanced.jsonl").resolve())
    val_jsonl = str((CURRENT_DIR.parent / "data" / "NMRGym_val_balanced.jsonl").resolve())
    test_jsonl = str((CURRENT_DIR.parent / "data" / "NMRGym_test_balanced.jsonl").resolve())

    print("Testing dataloader...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        test_jsonl=test_jsonl,
        batch_size=32,
        num_workers=0,
        max_peaks=200,
        augment_train=True
    )

    # Test batch
    batch = next(iter(train_loader))
    print(f"\n✓ Test batch:")
    print(f"  shifts: {batch['shifts'].shape}")
    print(f"  counts: {batch['counts'].shape}")
    print(f"  types: {batch['types'].shape}")
    print(f"  padding_mask: {batch['padding_mask'].shape}")
    print(f"  fg_labels: {batch['fg_labels'].shape}")

    # Compute class weights
    pos_weight, pos_counts = compute_class_weights(train_loader, num_classes=22)
    print(f"\n✓ Class weights computed: {pos_weight.shape}")
