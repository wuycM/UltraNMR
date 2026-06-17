"""
Training script for NMR superclass classification.

Loads a pretrained NMR encoder and trains a classification head
to predict chemical superclasses.

Usage:
    python train_classification.py --config configs/config_classification.json
"""

import argparse
import json
import os
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from models.nmr_classifier import build_nmr_classifier
from utils.nmr_classification_dataset import (
    NMRClassificationDataset,
    nmr_classification_collate_fn,
    build_class_mapping,
    save_class_mapping,
    load_class_mapping,
)


def setup_logger(save_dir):
    """Setup logger."""
    log_file = os.path.join(save_dir, f'training_classification_{datetime.now():%Y-%m-%d_%H-%M-%S}.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, config, writer, global_step):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    use_bf16 = config.get('use_bf16', False)
    max_grad_norm = config.get('max_grad_norm', 1.0)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        shifts = batch["raw_shifts"].to(device)
        counts = batch["counts"].to(device)
        types = batch["types"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = shifts.shape[0]

        optimizer.zero_grad()

        # Forward pass with mixed precision if enabled
        if use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(
                    shifts=shifts,
                    counts=counts,
                    types=types,
                    padding_mask=padding_mask
                )

                # Compute loss
                loss = F.cross_entropy(logits, labels)

            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf detected in loss at step {global_step}, skipping batch")
                optimizer.zero_grad()
                continue

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping
            if max_grad_norm is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    logging.warning(f"NaN/Inf detected in gradients at step {global_step}, skipping optimizer step")
                    optimizer.zero_grad()
                    scaler.update()
                    continue

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(
                shifts=shifts,
                counts=counts,
                types=types,
                padding_mask=padding_mask
            )

            # Compute loss
            loss = F.cross_entropy(logits, labels)

            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf detected in loss at step {global_step}, skipping batch")
                optimizer.zero_grad()
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            if max_grad_norm is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    logging.warning(f"NaN/Inf detected in gradients at step {global_step}, skipping optimizer step")
                    optimizer.zero_grad()
                    continue

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Calculate accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += batch_size

        # Update statistics
        loss_value = loss.item()
        total_loss += loss_value
        num_batches += 1

        # Logging
        if global_step % 10 == 0:
            writer.add_scalar('Train/loss', loss_value, global_step)
            if scheduler is not None:
                writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], global_step)
            if total_samples > 0:
                acc = total_correct / total_samples
                writer.add_scalar('Train/accuracy', acc, global_step)

        # Update progress bar
        acc = total_correct / total_samples if total_samples > 0 else 0.0
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
            'acc': f'{acc:.4f}',
        })

        global_step += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0

    return avg_loss, avg_acc, global_step


@torch.no_grad()
def validate(model, val_loader, device, writer, global_step):
    """Validate the model."""
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    # For per-class metrics
    all_predictions = []
    all_labels = []

    pbar = tqdm(val_loader, desc="Validation")

    for batch in pbar:
        shifts = batch["raw_shifts"].to(device)
        counts = batch["counts"].to(device)
        types = batch["types"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = shifts.shape[0]

        # Forward pass
        logits = model(
            shifts=shifts,
            counts=counts,
            types=types,
            padding_mask=padding_mask
        )

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()

        total_correct += correct
        total_samples += batch_size
        total_loss += loss.item()
        num_batches += 1

        # Store for per-class metrics
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # Update progress bar
        current_acc = total_correct / total_samples
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}',
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0

    # Log to tensorboard
    writer.add_scalar('Val/loss', avg_loss, global_step)
    writer.add_scalar('Val/accuracy', avg_acc, global_step)

    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'total_samples': total_samples,
        'predictions': all_predictions,
        'labels': all_labels,
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, config, save_path, val_metrics=None):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'global_step': global_step,
        'config': config,
    }

    if val_metrics is not None:
        checkpoint.update({
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
        })

    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train NMR superclass classifier")
    parser.add_argument('--config', type=str, default="../configs/config_classification.json", help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Set random seeds
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    device_id = config.get('device', 'cuda:0')
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create save directory
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Setup logger
    setup_logger(save_dir)

    # Save config
    config_save_path = os.path.join(save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f"Config saved to {config_save_path}")

    # Log configuration
    logging.info("=" * 60)
    logging.info("NMR Classification Training Configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 60)

    # Build or load class mapping
    class_mapping_path = os.path.join(save_dir, 'class_mapping.json')
    if os.path.exists(class_mapping_path):
        logging.info(f"Loading existing class mapping from {class_mapping_path}")
        class_to_idx, idx_to_class, class_counts = load_class_mapping(class_mapping_path)
    else:
        logging.info("Building class mapping from training data...")
        class_to_idx, idx_to_class, class_counts = build_class_mapping(
            config['train_data'],
            exclude_classes=config.get('exclude_classes', ['Failed'])
        )
        save_class_mapping(class_to_idx, idx_to_class, class_counts, class_mapping_path)

    num_classes = len(class_to_idx)
    logging.info(f"Number of classes: {num_classes}")

    # Build model
    logging.info("Building model...")
    model = build_nmr_classifier(
        config = config,
        pretrained_encoder_path=config['pretrained_encoder'],
        num_classes=num_classes,
        head_type=config.get('head_type', 'linear'),
        hidden_dim=config.get('hidden_dim', 512),
        dropout=config.get('dropout', 0.1),
        freeze_encoder=config.get('freeze_encoder', True),
        pooling=config.get('pooling', 'mean'),
        encoder_config=None,  # Will be loaded from checkpoint
        device=device
    )

    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = NMRClassificationDataset(
        jsonl_path=config['train_data'],
        class_to_idx=class_to_idx,
        split='train',
        h_shift_range=config.get('h_shift_range', (-0.01, 0.01)),
        c_shift_range=config.get('c_shift_range', (-0.1, 0.1)),
        shift_aug_p=config.get('shift_aug_p', 0.2),
        filter_unseen_classes=True,
    )

    val_dataset = NMRClassificationDataset(
        jsonl_path=config['val_data'],
        class_to_idx=class_to_idx,
        split='val',
        filter_unseen_classes=True,
    )

    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=nmr_classification_collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', config['batch_size']),
        shuffle=False,
        collate_fn=nmr_classification_collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
    )

    # Setup learning rate scheduler
    num_epochs = config['num_epochs']
    num_training_steps = len(train_loader) * num_epochs
    n_warmup_steps = config.get('n_warmup_steps', 0)

    if n_warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = None

    # Setup gradient scaler for bf16
    scaler = torch.cuda.amp.GradScaler() if config.get('use_bf16', False) else None

    # Setup tensorboard
    log_dir = os.path.join('runs', Path(save_dir).name)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logs: {log_dir}")

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0

    if args.resume or config.get('resume_from_checkpoint'):
        resume_path = args.resume or config.get('resume_from_checkpoint')
        logging.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_acc = checkpoint.get('val_accuracy', 0.0)
        logging.info(f"  Resumed from epoch {checkpoint['epoch']}, global step {global_step}")
        logging.info(f"  Best val acc so far: {best_val_acc:.4f}")

    # Training loop
    logging.info("\nStarting training...")
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"\nEpoch {epoch}/{num_epochs-1}")

        # Train
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, config, writer, global_step
        )

        logging.info(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, device, writer, global_step)

        logging.info(f"Val loss: {val_metrics['loss']:.4f}")
        logging.info(f"Val accuracy: {val_metrics['accuracy']:.4f}")

        # Save epoch checkpoint
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, global_step,
            config, checkpoint_path, val_metrics
        )

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_path = os.path.join(save_dir, 'model_best.pth')
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step,
                config, best_path, val_metrics
            )
            logging.info(f"New best model! Acc: {best_val_acc:.4f}")

    logging.info("\nTraining completed!")
    logging.info(f"Best validation accuracy: {best_val_acc:.4f}")

    writer.close()


if __name__ == '__main__':
    main()
