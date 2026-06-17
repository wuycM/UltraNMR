"""
Training script for NMR2SMILES model using NMRGym JSONL format data

NMRGym format:
    - h_shift: list of floats
    - c_shift: list of floats
    - smiles: target SMILES string

Usage:
    python train_nmrgym.py --config configs/config_nmrgym_train.json
"""

import argparse
import json
import os
import random
import logging
from functools import partial
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from models.UltraNMR import UltraNMR
from models.nmr2smiles import SmilesDecoder, NMR2SmilesModel
from utils.nmr2smiles_dataset import get_formula_counts
from utils.smiles_tokenizer import ChemBertaSmilesTokenizer
from layers.lora import apply_lora_to_transformer, count_lora_parameters

REPO_ROOT = Path(__file__).resolve().parent


def resolve_path(path_value, base_dir):
    if path_value is None or os.path.isabs(path_value):
        return path_value
    return str((base_dir / path_value).resolve())

class NMRGymDataset(Dataset):
    """
    Dataset for NMRGym JSONL data format.

    NMRGym JSONL format:
        - h_shift: list of floats (no count info, default to 1)
        - c_shift: list of floats
        - smiles: target SMILES string
    """

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_smiles_len=256,
        split='train',
        h_shift_range=(-0.01, 0.01),
        c_shift_range=(-0.1, 0.1),
        shift_aug_p=0.2,
        filter_invalid_smiles=True,
        h_min=0.01,
        h_max=16.0,
        c_min=0.01,
        c_max=230.0,
    ):
        self.tokenizer = tokenizer
        self.max_smiles_len = max_smiles_len
        self.split = split
        self.h_shift_range = h_shift_range
        self.c_shift_range = c_shift_range
        self.shift_aug_p = shift_aug_p
        self.filter_invalid_smiles = filter_invalid_smiles
        self.h_min = h_min
        self.h_max = h_max
        self.c_min = c_min
        self.c_max = c_max

        # Get special token ids
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        # Load data from JSONL (with filtering)
        print(f"Loading {split} data from {jsonl_path}...")
        self.data = []
        skipped_json = 0
        skipped_h_outliers = 0
        skipped_c_outliers = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    skipped_json += 1
                    continue

                # Filter out samples with NMR outliers
                h_shifts = item.get('h_shift', [])
                c_shifts = item.get('c_shift', [])

                # Check for H shift outliers
                has_h_outlier = any(h < h_min or h > h_max for h in h_shifts)
                if has_h_outlier:
                    skipped_h_outliers += 1
                    continue

                # Check for C shift outliers
                has_c_outlier = any(c < c_min or c > c_max for c in c_shifts)
                if has_c_outlier:
                    skipped_c_outliers += 1
                    continue

                # if self.filter_invalid_smiles:
                #     raw_smiles = item.get("smiles", "")
                #     canon = canonicalize_smiles(raw_smiles)
                #     if canon is None:
                #         skipped += 1
                #         continue
                #     item["smiles"] = canon

                self.data.append(item)

        total_skipped = skipped_json + skipped_h_outliers + skipped_c_outliers
        print(f"[{split}] Loaded {len(self.data)} samples")
        print(f"[{split}] Skipped: {total_skipped} total (JSON errors: {skipped_json}, H outliers: {skipped_h_outliers}, C outliers: {skipped_c_outliers})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # smiles is already guaranteed to be canonical and non-None here
        # because it was filtered in __init__.
        smiles = item["smiles"]

        # Parse NMR data
        h_shifts = item.get('h_shift', [])
        c_shifts = item.get('c_shift', [])

        # Convert to tensors
        h_shifts = torch.tensor(h_shifts, dtype=torch.float32)
        c_shifts = torch.tensor(c_shifts, dtype=torch.float32)

        #c_shifts = torch.round(c_shifts * 10) / 10
        # h_shifts = torch.unique(h_shifts)
        # c_shifts = torch.unique(c_shifts)

        # NMRGym doesn't have H count info, use 1 as default
        h_counts = torch.ones_like(h_shifts)
        c_counts = torch.zeros_like(c_shifts)

        # Shift augmentation (only for training)
        if self.split == 'train' and random.random() < self.shift_aug_p:
            h_bias = random.uniform(*self.h_shift_range)
            c_bias = random.uniform(*self.c_shift_range)
            h_shifts = h_shifts + h_bias
            c_shifts = c_shifts + c_bias

        # Build input sequence: [H peaks, C peaks]
        all_shifts = torch.cat([h_shifts, c_shifts])
        all_counts = torch.cat([h_counts, c_counts])
        types = torch.cat([
            torch.zeros_like(h_shifts, dtype=torch.long),  # H = 0
            torch.ones_like(c_shifts, dtype=torch.long)    # C = 1
        ])

        # Get formula counts from SMILES
        formula_counts = get_formula_counts(smiles)
        # Tokenize target SMILES
        token_ids = self.tokenizer.encode(smiles, add_special_tokens=False)
        smiles_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        if len(smiles_ids) > self.max_smiles_len:
            smiles_ids = smiles_ids[:self.max_smiles_len - 1] + [self.eos_token_id]
        smiles_ids = torch.tensor(smiles_ids, dtype=torch.long)

        return {
            "raw_shifts": all_shifts,
            "counts": all_counts,
            "types": types,
            "smiles_ids": smiles_ids,
            "smiles": smiles,
            "formula_counts": formula_counts,
        }


def nmrgym_collate_fn(batch, pad_idx=0):
    """Collate function for NMRGym training."""
    raw_shifts = [item['raw_shifts'] for item in batch]
    counts = [item['counts'] for item in batch]
    types = [item['types'] for item in batch]
    smiles_ids = [item['smiles_ids'] for item in batch]
    smiles_list = [item['smiles'] for item in batch]
    formula_counts = [item['formula_counts'] for item in batch]

    # Pad sequences
    raw_shifts_padded = pad_sequence(raw_shifts, batch_first=True, padding_value=0.0)
    counts_padded = pad_sequence(counts, batch_first=True, padding_value=0.0)
    types_padded = pad_sequence(types, batch_first=True, padding_value=-1)
    smiles_padded = pad_sequence(smiles_ids, batch_first=True, padding_value=pad_idx)
    formula_counts_stacked = torch.stack(formula_counts, dim=0)

    # Create padding masks
    nmr_padding_mask = (types_padded == -1)
    types_padded[nmr_padding_mask] = 0
    smiles_padding_mask = (smiles_padded == pad_idx)

    return {
        "raw_shifts": raw_shifts_padded,
        "counts": counts_padded,
        "types": types_padded,
        "nmr_padding_mask": nmr_padding_mask,
        "smiles_ids": smiles_padded,
        "smiles_padding_mask": smiles_padding_mask,
        "smiles_list": smiles_list,
        "formula_counts": formula_counts_stacked,
    }


def canonicalize_smiles(smiles: str) -> str:
    """Convert SMILES to canonical form."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
            #return Chem.MolToSmiles(mol, canonical=True,isomericSmiles=False)
    except:
        pass
    return None


def decode_smiles(token_ids, tokenizer, eos_idx):
    """Decode token IDs to SMILES string."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    if eos_idx in token_ids:
        eos_pos = token_ids.index(eos_idx)
        token_ids = token_ids[:eos_pos]

    smiles = tokenizer.decode(token_ids, skip_special_tokens=True)
    return smiles.strip()


def build_model(config, device):
    """Build NMR2SMILES model from config."""
    tokenizer = ChemBertaSmilesTokenizer(chemberta_path=config['tokenizer_path'])
    # Initialize encoder
    encoder = UltraNMR(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config.get('dropout', 0.1),
        h_bin_size=config.get('h_bin_size', 0.01),
        h_max=config.get('h_max', 16.0),
        c_bin_size=config.get('c_bin_size', 0.1),
        c_max=config.get('c_max', 230.0),
        use_identity_embedding=config.get('use_identity_embedding', True),
        use_count_embedding=config.get('use_count_embedding', False),
    )
    # Initialize decoder with ChemBERTa vocab
    vocab_size = len(tokenizer)
    print(f"Using ChemBERTa vocabulary (size={vocab_size})")

    decoder = SmilesDecoder(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        nhead=config.get('decoder_nhead', config['nhead']),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        dim_feedforward=config.get('decoder_dim_feedforward', config['dim_feedforward']),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_smiles_len', 256),
        pad_idx=tokenizer.pad_token_id,
        pretrained_embedding=None,  # Randomly initialize
        freeze_embedding=False,
    )

    # Create full model
    model = NMR2SmilesModel(
        encoder=encoder,
        decoder=decoder,
        freeze_encoder=config.get('freeze_encoder', False),
        use_count_fusion=config.get('use_count_fusion', True),
        use_formula_embedding=config.get('use_formula_embedding', True),
        num_atom_types=config.get('num_atom_types', 50),
    )

    if 'pretrained_encoder' in config and config['pretrained_encoder']:
        print(f"Loading pretrained encoder from: {config['pretrained_encoder']}")
        checkpoint = torch.load(config['pretrained_encoder'], map_location='cpu')
        model.encoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded encoder")

    # Load pretrained full model if specified
    if 'pretrained_model' in config and config['pretrained_model']:
        print(f"Loading pretrained model from: {config['pretrained_model']}")
        checkpoint = torch.load(config['pretrained_model'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"  Loaded complete model")

    # Apply LoRA if enabled
    if config.get('use_lora', False):
        print("\nApplying LoRA to model...")
        model = apply_lora_to_transformer(
            model=model,
            rank=config.get('lora_rank', 8),
            alpha=config.get('lora_alpha', 16.0),
            dropout=config.get('lora_dropout', 0.0),
            apply_to_encoder=config.get('lora_apply_to_encoder', True),
            apply_to_decoder=config.get('lora_apply_to_decoder', True),
        )
        num_lora_params = count_lora_parameters(model)
        print(f"  LoRA parameters: {num_lora_params / 1e6:.2f}M")

    model = model.to(device)
    return model, tokenizer


def setup_logger(save_dir):
    """Setup logger."""
    log_file = os.path.join(save_dir, f'training_nmrgym_{datetime.now():%Y-%m-%d_%H-%M-%S}.log')
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
    num_batches = 0
    total_correct_tokens = 0
    total_tokens = 0

    use_bf16 = config.get('use_bf16', False)
    label_smoothing = config.get('label_smoothing', 0.1)
    use_formula = config.get('use_formula_embedding', True)
    use_grad_clip = config.get('use_grad_clip', True)
    max_grad_norm = config.get('max_grad_norm', 1.0)

    # Gradient accumulation settings
    accumulation_steps = config.get('accumulation_steps', 1)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        shifts = batch["raw_shifts"].to(device)
        counts = batch["counts"].to(device)
        types = batch["types"].to(device)
        nmr_padding_mask = batch["nmr_padding_mask"].to(device)
        smiles_ids = batch["smiles_ids"].to(device)
        smiles_padding_mask = batch["smiles_padding_mask"].to(device)
        formula_counts = batch["formula_counts"].to(device) if use_formula else None

        # Teacher forcing: input is [BOS, ..., token_{n-1}], target is [token_0, ..., EOS]
        tgt_input = smiles_ids[:, :-1]
        tgt_output = smiles_ids[:, 1:]
        tgt_padding_mask = smiles_padding_mask[:, :-1]

        # Only zero gradients at the start of accumulation
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()

        # Forward pass with mixed precision if enabled
        if use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(
                    shifts=shifts,
                    counts=counts,
                    types=types,
                    nmr_padding_mask=nmr_padding_mask,
                    tgt_ids=tgt_input,
                    tgt_padding_mask=tgt_padding_mask,
                    formula_counts=formula_counts,
                )

                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1),
                    ignore_index=model.decoder.pad_idx,
                    label_smoothing=label_smoothing,
                )

            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf detected in loss at step {global_step}, skipping batch")
                if batch_idx % accumulation_steps == 0:
                    optimizer.zero_grad()
                continue

            # Scale loss by accumulation steps
            scaled_loss = loss / accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(scaled_loss).backward()

            # Only update weights after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)

                # Gradient clipping (optional)
                if use_grad_clip:
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
                nmr_padding_mask=nmr_padding_mask,
                tgt_ids=tgt_input,
                tgt_padding_mask=tgt_padding_mask,
                formula_counts=formula_counts,
            )

            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=model.decoder.pad_idx,
                label_smoothing=label_smoothing,
            )

            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf detected in loss at step {global_step}, skipping batch")
                if batch_idx % accumulation_steps == 0:
                    optimizer.zero_grad()
                continue

            # Scale loss by accumulation steps
            scaled_loss = loss / accumulation_steps

            # Backward pass
            scaled_loss.backward()

            # Only update weights after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping (optional)
                if use_grad_clip:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        logging.warning(f"NaN/Inf detected in gradients at step {global_step}, skipping optimizer step")
                        optimizer.zero_grad()
                        continue

                optimizer.step()

        # Update scheduler after each accumulation cycle
        if scheduler is not None and (batch_idx + 1) % accumulation_steps == 0:
            scheduler.step()

        # Calculate token-level accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            mask = (tgt_output != model.decoder.pad_idx)
            correct = (predictions == tgt_output) & mask
            batch_correct_tokens = correct.sum().item()
            batch_total_tokens = mask.sum().item()

            total_correct_tokens += batch_correct_tokens
            total_tokens += batch_total_tokens

        # Update statistics
        loss_value = loss.item()
        total_loss += loss_value
        num_batches += 1

        # Only log and increment global_step after accumulation cycle completes
        if (batch_idx + 1) % accumulation_steps == 0:
            # Logging
            if global_step % 10 == 0:
                writer.add_scalar('Train/loss', loss_value, global_step)
                if scheduler is not None:
                    writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], global_step)
                if total_tokens > 0:
                    token_acc = total_correct_tokens / total_tokens
                    writer.add_scalar('Train/token_acc', token_acc, global_step)

            global_step += 1

        # Update progress bar
        token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
            'token_acc': f'{token_acc:.4f}',
            'accum': f'{(batch_idx % accumulation_steps) + 1}/{accumulation_steps}',
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, global_step


@torch.no_grad()
def validate(model, val_loader, tokenizer, device, config, writer, global_step):
    """Validate the model."""
    model.eval()

    bos_idx = tokenizer.bos_token_id
    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id
    use_formula = config.get('use_formula_embedding', True)
    label_smoothing = config.get('label_smoothing', 0.1)

    total_loss = 0
    num_batches = 0
    canonical_correct = 0
    valid_generated = 0
    total_samples = 0

    pbar = tqdm(val_loader, desc="Validation")

    for batch in pbar:
        shifts = batch["raw_shifts"].to(device)
        counts = batch["counts"].to(device)
        types = batch["types"].to(device)
        nmr_padding_mask = batch["nmr_padding_mask"].to(device)
        smiles_ids = batch["smiles_ids"].to(device)
        smiles_padding_mask = batch["smiles_padding_mask"].to(device)
        smiles_list = batch["smiles_list"]
        formula_counts = batch["formula_counts"].to(device) if use_formula else None

        batch_size = shifts.shape[0]

        # Compute loss
        tgt_input = smiles_ids[:, :-1]
        tgt_output = smiles_ids[:, 1:]
        tgt_padding_mask = smiles_padding_mask[:, :-1]

        logits = model(
            shifts=shifts,
            counts=counts,
            types=types,
            nmr_padding_mask=nmr_padding_mask,
            tgt_ids=tgt_input,
            tgt_padding_mask=tgt_padding_mask,
            formula_counts=formula_counts,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
        )

        total_loss += loss.item()
        num_batches += 1

        # Generate SMILES for accuracy evaluation
        generated_ids = model.generate(
            shifts=shifts,
            counts=counts,
            types=types,
            nmr_padding_mask=nmr_padding_mask,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            max_len=config.get('max_smiles_len', 256),
            temperature=0,
            formula_counts=formula_counts,
        )

        # Evaluate accuracy
        for i in range(batch_size):
            total_samples += 1

            gen_smiles = decode_smiles(generated_ids[i, 1:], tokenizer, eos_idx)
            tgt_smiles = smiles_list[i]

            gen_canonical = canonicalize_smiles(gen_smiles)
            tgt_canonical = canonicalize_smiles(tgt_smiles)

            if gen_canonical is not None:
                valid_generated += 1

            if gen_canonical is not None and tgt_canonical is not None:
                if gen_canonical == tgt_canonical:
                    canonical_correct += 1

        # Update progress bar
        current_acc = canonical_correct / total_samples if total_samples > 0 else 0
        current_valid = valid_generated / total_samples if total_samples > 0 else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}',
            'valid': f'{current_valid:.4f}',
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    canonical_acc = canonical_correct / total_samples if total_samples > 0 else 0
    valid_rate = valid_generated / total_samples if total_samples > 0 else 0

    # Log to tensorboard
    writer.add_scalar('Val/loss', avg_loss, global_step)
    writer.add_scalar('Val/canonical_acc', canonical_acc, global_step)
    writer.add_scalar('Val/valid_rate', valid_rate, global_step)

    return {
        'loss': avg_loss,
        'canonical_acc': canonical_acc,
        'valid_rate': valid_rate,
        'total_samples': total_samples,
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
            'val_canonical_acc': val_metrics['canonical_acc'],
            'val_valid_rate': val_metrics['valid_rate'],
        })

    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train NMR2SMILES model with NMRGym data")
    parser.add_argument('--config', type=str, default='configs/config_nmrgym_train.json', help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config_path = config_path.resolve()
    with open(config_path, 'r') as f:
        config = json.load(f)

    config_base = config_path.parent
    for key in ['train_data', 'val_data', 'tokenizer_path', 'pretrained_encoder', 'pretrained_model', 'resume_from_checkpoint']:
        config[key] = resolve_path(config.get(key), config_base)

    if args.resume is not None:
        args.resume = resolve_path(args.resume, config_base)

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
    logging.info("NMRGym Training Configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 60)

    # Build model
    logging.info("Building model...")
    model, tokenizer = build_model(config, device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_encoder_params = sum(p.numel() for p in model.encoder.parameters())
    num_decoder_params = sum(p.numel() for p in model.decoder.parameters())

    logging.info(f"Total parameters: {num_params / 1e6:.2f}M")
    logging.info(f"  Encoder parameters: {num_encoder_params / 1e6:.2f}M")
    logging.info(f"  Decoder parameters: {num_decoder_params / 1e6:.2f}M")
    logging.info(f"Trainable parameters: {num_trainable / 1e6:.2f}M")
    logging.info(f"Encoder frozen: {config.get('freeze_encoder', False)}")
    if config.get('use_lora', False):
        num_lora_params = count_lora_parameters(model)
        logging.info(f"LoRA enabled: True")
        logging.info(f"  LoRA rank: {config.get('lora_rank', 8)}")
        logging.info(f"  LoRA alpha: {config.get('lora_alpha', 16.0)}")
        logging.info(f"  LoRA parameters: {num_lora_params / 1e6:.2f}M ({num_lora_params / num_params * 100:.2f}% of total)")
    else:
        logging.info(f"LoRA enabled: False")

    # Log training settings
    logging.info("\nTraining Settings:")
    logging.info(f"  BF16 training: {config.get('use_bf16', False)}")
    logging.info(f"  Label smoothing: {config.get('label_smoothing', 0.1)}")
    logging.info(f"  Gradient clipping: {config.get('use_grad_clip', True)}")
    if config.get('use_grad_clip', True):
        logging.info(f"  Max gradient norm: {config.get('max_grad_norm', 1.0)}")

    # Create datasets
    logging.info("\nCreating datasets...")
    train_dataset = NMRGymDataset(
        jsonl_path=config['train_data'],
        tokenizer=tokenizer,
        max_smiles_len=config.get('max_smiles_len', 256),
        split='train',
        h_shift_range=config.get('h_shift_range', (-0.01, 0.01)),
        c_shift_range=config.get('c_shift_range', (-0.1, 0.1)),
        shift_aug_p=config.get('shift_aug_p', 0.2),
    )

    val_dataset = NMRGymDataset(
        jsonl_path=config['val_data'],
        tokenizer=tokenizer,
        max_smiles_len=config.get('max_smiles_len', 256),
        split='val',
    )
    val_dataset = Subset(val_dataset, indices=range(0, 2708))

    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=partial(nmrgym_collate_fn, pad_idx=tokenizer.pad_token_id),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', config['batch_size']),
        shuffle=False,
        collate_fn=partial(nmrgym_collate_fn, pad_idx=tokenizer.pad_token_id),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    # Setup optimizer - only optimize parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.get('weight_decay', 0.01),
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    optimizer = torch.optim.AdamW(
        #trainable_params,
        optimizer_grouped_parameters,
        lr=config['learning_rate'],
        #weight_decay=config.get('weight_decay', 0.01),
    )

    # Setup learning rate scheduler
    num_epochs = config['num_epochs']
    use_scheduler = config.get('use_scheduler', True)

    if use_scheduler:
        num_warmup_steps = 2 * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=len(train_loader) * num_epochs,
        )
        logging.info(f"Using cosine scheduler with {num_warmup_steps} warmup steps")
    else:
        scheduler = None
        logging.info("Using constant learning rate (no scheduler)")
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
        best_val_acc = checkpoint.get('val_canonical_acc', 0.0)
        logging.info(f"  Resumed from epoch {checkpoint['epoch']}, global step {global_step}")
        logging.info(f"  Best val acc so far: {best_val_acc:.4f}")

    # Training loop
    logging.info("\nStarting training...")
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"\nEpoch {epoch}/{num_epochs-1}")

        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, config, writer, global_step
        )

        logging.info(f"Train loss: {train_loss:.4f}")
                
        #if epoch % 5 == 0 and epoch != 0:
        if epoch % 1 == 0:
            # Save epoch checkpoint
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step,
                config, checkpoint_path
            )
            # Validate
            val_metrics = validate(
                model, val_loader, tokenizer, device, config,
                writer, global_step
            )

            logging.info(f"Val loss: {val_metrics['loss']:.4f}")
            logging.info(f"Val canonical acc: {val_metrics['canonical_acc']:.4f}")
            logging.info(f"Val valid rate: {val_metrics['valid_rate']:.4f}")
            # Save best model
            if val_metrics['canonical_acc'] > best_val_acc:
                best_val_acc = val_metrics['canonical_acc']
                best_path = os.path.join(save_dir, 'model_best.pth')
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, global_step,
                    config, best_path, val_metrics
                )
                logging.info(f"New best model! Acc: {best_val_acc:.4f}")

    logging.info("\nTraining completed!")
    logging.info(f"Best validation accuracy: {best_val_acc:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
