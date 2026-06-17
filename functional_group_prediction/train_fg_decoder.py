"""
Training script for Functional Group Decoder
Finetune UltraNMR encoder on NMRGym balanced data
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import argparse
import yaml
import os
import json
from datetime import datetime
from tqdm import tqdm
import sys
import random
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
FALLBACK_FUNCG_DIR = REPO_ROOT.parent / "NMR-foundation" / "functional_group_code"

for path in (CURRENT_DIR, REPO_ROOT, FALLBACK_FUNCG_DIR):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.append(path_str)

from fg_decoder import create_fg_decoder
from fg_dataloader import create_dataloaders, compute_class_weights
from fg_loss_metrics import (
    FGLoss, evaluate_model, print_metrics, print_per_class_metrics
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


def set_seed(seed=42):
    """
    Set random seed for reproducibility

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ Random seed set to {seed} for reproducibility")


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, scheduler=None, scheduler_step_mode=None, scaler=None, use_amp=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        shifts = batch['shifts'].to(device)
        counts = batch['counts'].to(device)
        types = batch['types'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        fg_labels = batch['fg_labels'].to(device)

        optimizer.zero_grad()

        # Forward pass with AMP
        if use_amp:
            with autocast(dtype=torch.bfloat16):
                outputs = model(shifts, counts, types, padding_mask)
                loss = loss_fn(outputs, fg_labels)

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without AMP
            outputs = model(shifts, counts, types, padding_mask)
            loss = loss_fn(outputs, fg_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Step scheduler if mode is 'step' (for warmup_cosine)
        if scheduler is not None and scheduler_step_mode == 'step':
            scheduler.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, dataloader, loss_fn, device, epoch, config):
    """Validate for one epoch with full metrics evaluation"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Validation/Inference always uses FP32 for accuracy
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            shifts = batch['shifts'].to(device)
            counts = batch['counts'].to(device)
            types = batch['types'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            fg_labels = batch['fg_labels'].to(device)

            # Forward pass in FP32 (no AMP for inference)
            outputs = model(shifts, counts, types, padding_mask)
            loss = loss_fn(outputs, fg_labels)

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches

    # Evaluate full metrics on validation set (also in FP32)
    val_metrics, val_per_class = evaluate_model(
        model,
        dataloader,
        device,
        use_optimal_thresholds=config['evaluation'].get('use_optimal_thresholds', True),
        threshold=config['evaluation'].get('threshold', 0.5),
        apply_sigmoid=config['training']['loss'].get('use_logits', True)
    )

    return avg_loss, val_metrics, val_per_class


def save_checkpoint(model, optimizer, epoch, loss, val_metrics, save_dir, is_best=False):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_metrics': val_metrics
    }

    # Save latest
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)

    # Save best
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"✓ Best model saved: {best_path} (Val F1: {val_metrics['f1_macro']:.4f})")

    # Save periodic checkpoints
    if epoch % 10 == 0:
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['loss']
    best_val_f1 = checkpoint.get('val_metrics', {}).get('f1_macro', 0.0)

    print(f"✓ Resumed from epoch {checkpoint['epoch']}")
    print(f"  Previous best val loss: {best_val_loss:.4f}")
    print(f"  Previous best val F1: {best_val_f1:.4f}")

    return start_epoch, best_val_loss, best_val_f1


def main(config_path, resume_checkpoint=None):
    """Main training function"""
    config_path = resolve_local_path(config_path)
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint).expanduser()
        if not resume_checkpoint.is_absolute():
            resume_checkpoint = (config_path.parent / resume_checkpoint).resolve()
        resume_checkpoint = str(resume_checkpoint)

    # Load config
    with open(config_path, 'r') as f:
        if config_path.suffix in {'.yaml', '.yml'}:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    for section, key in [
        ('data', 'train_jsonl'),
        ('data', 'val_jsonl'),
        ('data', 'test_jsonl'),
        ('model', 'pretrained_checkpoint'),
        ('training', 'save_dir'),
    ]:
        value = config.get(section, {}).get(key)
        if isinstance(value, str) and not os.path.isabs(value):
            config[section][key] = str((config_path.parent / value).resolve())

    print("\n" + "=" * 60)
    print("Functional Group Decoder Training")
    print("=" * 60)

    # Set random seed for reproducibility
    seed = config.get('seed', 0)
    set_seed(seed)

    # Setup device
    device = select_device()
    print(f"Device: {device}")

    # Setup mixed precision training
    use_amp = config['training'].get('use_amp', False)
    scaler = None
    if use_amp:
        if not torch.cuda.is_available():
            print("⚠ Warning: AMP requested but CUDA not available. Disabling AMP.")
            use_amp = False
        else:
            # Check if bf16 is supported
            if torch.cuda.is_bf16_supported():
                scaler = GradScaler()  # GradScaler auto-detects CUDA device
                print(f"✓ Using BF16 mixed precision for training")
                print(f"✓ Using FP32 for validation/testing (inference)")
            else:
                print("⚠ Warning: BF16 not supported on this GPU. Falling back to FP32.")
                use_amp = False
    else:
        print("✗ Using standard FP32 for both training and inference")

    # Create save directory
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")

    # Save config
    config_save_path = os.path.join(save_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Config saved: {config_save_path}")

    # Create dataloaders
    print("\n" + "=" * 60)
    print("Creating dataloaders...")
    print("=" * 60)

    # Get exclude_classes from config
    exclude_classes = config['data'].get('exclude_classes', None)
    if exclude_classes:
        print(f"\n⚠ Warning: Classes {exclude_classes} will be excluded from training and evaluation")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_jsonl=config['data']['train_jsonl'],
        val_jsonl=config['data']['val_jsonl'],
        test_jsonl=config['data']['test_jsonl'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        max_peaks=config['data'].get('max_peaks', 200),
        augment_train=config['data'].get('augment', True),
        noise_std=config['data'].get('noise_std', 0.02),
        scale_range=tuple(config['data'].get('scale_range', [0.95, 1.05])),
        exclude_classes=exclude_classes
    )

    # Compute class weights if needed
    pos_weight = None
    cls_num_list = None
    loss_strategy = config['training']['loss']['strategy']

    if loss_strategy in ['weighted', 'ldam']:
        print("\n" + "=" * 60)
        print(f"Computing class weights for {loss_strategy}")
        print("=" * 60)
        pos_weight, cls_num_list = compute_class_weights(
            train_loader,
            num_classes=config['model']['num_functional_groups'],
            device=device,
            exclude_classes=exclude_classes
        )

    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)

    # Merge encoder_config with model config
    encoder_config = config['model'].get('encoder_config', {})
    full_config = {**encoder_config, **config['model']}  # Model config overrides encoder_config

    model = create_fg_decoder(
        checkpoint_path=config['model']['pretrained_checkpoint'],
        config=full_config,
        num_functional_groups=config['model']['num_functional_groups'],
        freeze_encoder=config['model'].get('freeze_encoder', False),
        use_logits=config['training']['loss'].get('use_logits', True),
        device=device
    )

    # Create loss function
    print("\n" + "=" * 60)
    print("Creating loss function...")
    print("=" * 60)

    # Check if using per-class focal loss parameters
    use_per_class_params = config['training']['loss'].get('use_per_class_params', False)
    focal_alpha_per_class = None
    focal_gamma_per_class = None

    if use_per_class_params and loss_strategy == 'focal':
        focal_alpha_per_class = config['training']['loss'].get('focal_alpha_per_class', None)
        focal_gamma_per_class = config['training']['loss'].get('focal_gamma_per_class', None)

        # Validate lengths
        num_classes = config['model']['num_functional_groups']
        if focal_alpha_per_class is not None and len(focal_alpha_per_class) != num_classes:
            raise ValueError(f"focal_alpha_per_class must have {num_classes} values, got {len(focal_alpha_per_class)}")
        if focal_gamma_per_class is not None and len(focal_gamma_per_class) != num_classes:
            raise ValueError(f"focal_gamma_per_class must have {num_classes} values, got {len(focal_gamma_per_class)}")

        print(f"✓ Using per-class focal loss parameters")

    loss_fn = FGLoss(
        strategy=loss_strategy,
        pos_weight=pos_weight,
        focal_alpha=config['training']['loss'].get('focal_alpha', 0.25),
        focal_gamma=config['training']['loss'].get('focal_gamma', 2.0),
        focal_alpha_per_class=focal_alpha_per_class,
        focal_gamma_per_class=focal_gamma_per_class,
        cls_num_list=cls_num_list,
        ldam_max_m=config['training']['loss'].get('ldam_max_m', 0.5),
        ldam_s=config['training']['loss'].get('ldam_s', 30),
        use_logits=config['training']['loss'].get('use_logits', True),
        label_smoothing=config['training']['loss'].get('label_smoothing', 0.0)
    )

    # Create optimizer with differential learning rates
    print("\n" + "=" * 60)
    print("Creating optimizer with differential learning rates...")
    print("=" * 60)

    # Check if using differential learning rates
    use_diff_lr = config['training']['optimizer'].get('use_differential_lr', False)

    if use_diff_lr:
        # Get encoder and decoder learning rates
        encoder_lr = config['training']['optimizer'].get('encoder_lr', 5e-6)
        decoder_lr = config['training']['optimizer'].get('decoder_lr', 1e-4)
        encoder_weight_decay = config['training']['optimizer'].get('encoder_weight_decay', 0.01)
        decoder_weight_decay = config['training']['optimizer'].get('decoder_weight_decay', 0.05)

        # Separate encoder and decoder parameters
        encoder_params = []
        decoder_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Encoder parameters
            if 'encoder' in name:
                encoder_params.append(param)
            # Decoder parameters (classifier, attention_weights, etc.)
            else:
                decoder_params.append(param)

        # Create parameter groups
        param_groups = [
            {
                'params': encoder_params,
                'lr': encoder_lr,
                'weight_decay': encoder_weight_decay,
                'name': 'encoder'
            },
            {
                'params': decoder_params,
                'lr': decoder_lr,
                'weight_decay': decoder_weight_decay,
                'name': 'decoder'
            }
        ]

        optimizer = torch.optim.AdamW(param_groups)

        print(f"✓ Using differential learning rates:")
        print(f"  - Encoder: lr={encoder_lr:.2e}, weight_decay={encoder_weight_decay}")
        print(f"  - Decoder: lr={decoder_lr:.2e}, weight_decay={decoder_weight_decay}")
        print(f"  - LR ratio (decoder/encoder): {decoder_lr/encoder_lr:.1f}x")
        print(f"  - Encoder params: {len(encoder_params)}")
        print(f"  - Decoder params: {len(decoder_params)}")
    else:
        # Use uniform learning rate (original behavior)
        lr = config['training']['optimizer'].get('lr', 5e-5)
        weight_decay = config['training']['optimizer'].get('weight_decay', 0.01)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        print(f"✓ Using uniform learning rate:")
        print(f"  - LR: {lr:.2e}")
        print(f"  - Weight decay: {weight_decay}")

    # Create learning rate scheduler
    scheduler_type = config['training']['optimizer'].get('scheduler_type', 'plateau')

    if scheduler_type == 'warmup_cosine':
        # Warmup + Cosine Annealing scheduler
        warmup_epochs = config['training']['optimizer'].get('warmup_epochs', 2)
        num_epochs = config['training']['num_epochs']
        min_lr_ratio = config['training']['optimizer'].get('min_lr_ratio', 0.1)

        # Calculate steps per epoch
        steps_per_epoch = len(train_loader)
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = num_epochs * steps_per_epoch

        # Lambda function for warmup + cosine
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_ratio, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler_step_mode = 'step'  # Step every batch

        print(f"✓ Using Warmup + Cosine Annealing scheduler:")
        print(f"  - Warmup epochs: {warmup_epochs}")
        print(f"  - Total epochs: {num_epochs}")
        print(f"  - Min LR ratio: {min_lr_ratio}")
        print(f"  - Warmup steps: {warmup_steps}")
        print(f"  - Total steps: {total_steps}")

    elif scheduler_type == 'plateau':
        # ReduceLROnPlateau scheduler (original)
        factor = config['training']['optimizer'].get('scheduler_factor', 0.5)
        patience = config['training']['optimizer'].get('scheduler_patience', 5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience
        )
        scheduler_step_mode = 'epoch'  # Step every epoch based on val_loss

        print(f"✓ Using ReduceLROnPlateau scheduler:")
        print(f"  - Factor: {factor}")
        print(f"  - Patience: {patience} epochs")

    else:
        scheduler = None
        scheduler_step_mode = None
        print("✗ No scheduler used")

    # TensorBoard
    log_dir = os.path.join(save_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)
    print(f"✓ TensorBoard log: {log_dir}")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_f1 = 0.0

    if resume_checkpoint is not None:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        start_epoch, best_val_loss, best_val_f1 = load_checkpoint(model, optimizer, resume_checkpoint)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    num_epochs = config['training']['num_epochs']

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            scheduler=scheduler, scheduler_step_mode=scheduler_step_mode,
            scaler=scaler, use_amp=use_amp
        )
        print(f"Train Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Validate with full metrics (always in FP32 for accuracy)
        val_loss, val_metrics, val_per_class = validate_epoch(
            model, val_loader, loss_fn, device, epoch, config
        )
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Macro F1: {val_metrics['f1_macro']:.4f}")
        print(f"Val Macro Precision: {val_metrics['precision_macro']:.4f}")
        print(f"Val Macro Recall: {val_metrics['recall_macro']:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)

        # Learning rate scheduler (only step if mode is 'epoch' for ReduceLROnPlateau)
        if scheduler is not None and scheduler_step_mode == 'epoch':
            scheduler.step(val_loss)

        # Log learning rates
        if use_diff_lr:
            encoder_lr = optimizer.param_groups[0]['lr']
            decoder_lr = optimizer.param_groups[1]['lr']
            print(f"Current LR - Encoder: {encoder_lr:.2e}, Decoder: {decoder_lr:.2e}")
            writer.add_scalar('Learning_rate/encoder', encoder_lr, epoch)
            writer.add_scalar('Learning_rate/decoder', decoder_lr, epoch)
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.2e}")
            writer.add_scalar('Learning_rate', current_lr, epoch)

        # Save checkpoint based on best validation F1 score
        is_best = val_metrics['f1_macro'] > best_val_f1
        save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, save_dir, is_best)

        if is_best:
            best_val_f1 = val_metrics['f1_macro']
            best_val_loss = val_loss
            print(f"✓ New best model! Val F1: {best_val_f1:.4f}, Val Loss: {best_val_loss:.4f}")

        # Save validation metrics
        val_metrics_path = os.path.join(save_dir, f'val_metrics_epoch_{epoch}.json')
        val_metrics_to_save = {
            'overall': val_metrics,
            'per_class': val_per_class
        }
        with open(val_metrics_path, 'w') as f:
            json.dump(val_metrics_to_save, f, indent=2)

        # Evaluate on test set periodically
        if epoch % config['training'].get('eval_every', 10) == 0:
            print(f"\n{'=' * 60}")
            print(f"Evaluating on test set (Epoch {epoch})...")
            print(f"{'=' * 60}")

            test_metrics, test_per_class = evaluate_model(
                model,
                test_loader,
                device,
                use_optimal_thresholds=config['evaluation'].get('use_optimal_thresholds', True),
                threshold=config['evaluation'].get('threshold', 0.5),
                apply_sigmoid=config['training']['loss'].get('use_logits', True)
            )

            print_metrics(test_metrics, title=f"Test Metrics (Epoch {epoch})")
            print_per_class_metrics(test_per_class, sort_by='f1', top_n=10)

            # Log to TensorBoard
            for metric_name, metric_value in test_metrics.items():
                writer.add_scalar(f'Test/{metric_name}', metric_value, epoch)

            # Save metrics
            metrics_path = os.path.join(save_dir, f'test_metrics_epoch_{epoch}.json')
            metrics_to_save = {
                'overall': test_metrics,
                'per_class': test_per_class
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"✓ Test metrics saved: {metrics_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final evaluation on test set...")
    print("=" * 60)

    # Load best model
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")

    test_metrics, test_per_class = evaluate_model(
        model,
        test_loader,
        device,
        use_optimal_thresholds=config['evaluation'].get('use_optimal_thresholds', True),
        threshold=config['evaluation'].get('threshold', 0.5),
        apply_sigmoid=config['training']['loss'].get('use_logits', True)
    )

    print_metrics(test_metrics, title="Final Test Metrics")
    print_per_class_metrics(test_per_class, sort_by='f1', top_n=22)

    # Save final metrics
    final_metrics_path = os.path.join(save_dir, 'final_metrics.json')
    final_metrics = {
        'overall': test_metrics,
        'per_class': test_per_class
    }
    with open(final_metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n✓ Final metrics saved: {final_metrics_path}")

    writer.close()
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Functional Group Decoder')
    parser.add_argument('--config', type=str, default='config.yaml',help='Path to config file (YAML or JSON)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    main(args.config, args.resume)
