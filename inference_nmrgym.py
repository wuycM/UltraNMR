"""
Inference script for NMR2SMILES Model on NMRGym test dataset.

Usage:
    python inference_nmrgym.py \
        --checkpoint checkpoints/model_best.pth \
        --test_data data/NMRGym_test_balanced.jsonl
"""

import argparse
import pickle
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from functools import partial
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from models.UltraNMR import UltraNMR
from models.nmr2smiles import SmilesDecoder, NMR2SmilesModel
from utils.nmr2smiles_dataset import get_formula_counts
from utils.smiles_tokenizer import BasicSmilesTokenizer, ChemBertaSmilesTokenizer

REPO_ROOT = Path(__file__).resolve().parent


def resolve_repo_path(path_value):
    if path_value is None or os.path.isabs(path_value):
        return path_value
    return str((REPO_ROOT / path_value).resolve())


class NMRGymTestDataset(Dataset):
    """
    Dataset for NMRGym test data.

    NMRGym format:
        - h_shift: list of floats (no count info)
        - c_shift: list of floats
        - smiles: target SMILES
    """

    def __init__(self, pkl_path: str, tokenizer, max_smiles_len: int = 256):
        # with open(pkl_path, 'rb') as f:
        #     self.data = pickle.load(f)

        self.data = []
        with open(pkl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                self.data.append(item)

        self.tokenizer = tokenizer
        self.max_smiles_len = max_smiles_len
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        smiles = item['smiles']
        h_shifts = item.get('h_shift', [])
        c_shifts = item.get('c_shift', [])

        # Convert to tensors
        h_shifts = torch.tensor(h_shifts, dtype=torch.float32)
        c_shifts = torch.tensor(c_shifts, dtype=torch.float32)

        #c_shifts = torch.round(c_shifts * 10) / 10
        # h_shifts =  torch.unique(h_shifts)
        # c_shifts =  torch.unique(c_shifts)

        # H counts not available in NMRGym, use 1 as default
        h_counts = torch.ones_like(h_shifts)
        c_counts = torch.zeros_like(c_shifts)

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
    """Collate function for NMRGym test dataset."""
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

    return {
        "raw_shifts": raw_shifts_padded,
        "counts": counts_padded,
        "types": types_padded,
        "nmr_padding_mask": nmr_padding_mask,
        "smiles_ids": smiles_padded,
        "smiles_list": smiles_list,
        "formula_counts": formula_counts_stacked,
    }


def canonicalize_smiles(smiles: str, consider_stereochemistry: bool = False) -> str:
    """Convert SMILES to canonical form."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            if consider_stereochemistry:
                return Chem.MolToSmiles(mol, canonical=True)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        pass
    return None


def compute_tanimoto_similarity(smiles1: str, smiles2: str, radius: int = 2, n_bits: int = 2048) -> float:
    """
    Compute Tanimoto similarity between two molecules using Morgan fingerprints.

    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        radius: Morgan fingerprint radius (default: 2, equivalent to ECFP4)
        n_bits: Number of bits in fingerprint (default: 2048)

    Returns:
        Tanimoto similarity score (0.0 to 1.0), or None if either SMILES is invalid
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return None

        # Generate Morgan fingerprints (ECFP-like)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)

        # Compute Tanimoto similarity
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
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


def load_model(checkpoint_path, config, device):
    """Load model from checkpoint."""
    # Support multiple tokenizer types for proper stereochemistry handling
    tokenizer_type = config.get('tokenizer_type', 'chemberta')  # "chemberta" or "basic"

    if tokenizer_type == 'chemberta':
        chemberta_path = config.get('tokenizer_path', 'Chemberta_ckpt')
        tokenizer = ChemBertaSmilesTokenizer(chemberta_path=chemberta_path)
        print(f"Using ChemBertaSmilesTokenizer (vocab size: {len(tokenizer)})")
    else:
        vocab_file = config.get('vocab_file', None)
        if vocab_file:
            tokenizer = BasicSmilesTokenizer(vocab_file=vocab_file)
        else:
            tokenizer = BasicSmilesTokenizer()
        print(f"Using BasicSmilesTokenizer (vocab size: {len(tokenizer)})")

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

    # Get vocab size from tokenizer
    vocab_size = len(tokenizer)

    # Initialize decoder
    decoder = SmilesDecoder(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        nhead=config.get("decoder_nhead", config["nhead"]),
        num_decoder_layers=config.get("num_decoder_layers", 6),
        dim_feedforward=config.get("decoder_dim_feedforward", config["dim_feedforward"]),
        dropout=config["dropout"],
        max_seq_len=config.get("max_smiles_len", 256),
        pad_idx=tokenizer.pad_token_id,
        pretrained_embedding=None,
        freeze_embedding=config.get("freeze_embedding", False),
    )

    # Create model
    model = NMR2SmilesModel(
        encoder=encoder,
        decoder=decoder,
        freeze_encoder=False,
        use_count_fusion=config.get('use_count_fusion', False),
        use_formula_embedding=config.get('use_formula_embedding', False),
        num_atom_types=config.get('num_atom_types', 50),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def run_inference(
    model,
    dataloader,
    tokenizer,
    device,
    max_gen_len=256,
    use_formula_embedding=False,
    consider_stereochemistry=False,
):
    """Run inference on test dataset."""
    bos_idx = tokenizer.bos_token_id
    eos_idx = tokenizer.eos_token_id

    total_samples = 0
    canonical_correct = 0
    valid_generated = 0
    tanimoto_scores = []  # Store all valid Tanimoto scores
    results = []

    pbar = tqdm(dataloader, desc="Inference")

    for batch in pbar:
        shifts = batch["raw_shifts"].to(device)
        counts = batch["counts"].to(device)
        types = batch["types"].to(device)
        nmr_padding_mask = batch["nmr_padding_mask"].to(device)
        smiles_list = batch["smiles_list"]
        formula_counts = batch["formula_counts"].to(device) if use_formula_embedding else None

        batch_size = shifts.shape[0]

        # Generate SMILES
        generated_ids = model.generate(
            shifts=shifts,
            counts=counts,
            types=types,
            nmr_padding_mask=nmr_padding_mask,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            max_len=max_gen_len,
            temperature=0,
            formula_counts=formula_counts,
        )

        for i in range(batch_size):
            total_samples += 1

            gen_smiles = decode_smiles(generated_ids[i, 1:], tokenizer, eos_idx)
            tgt_smiles = smiles_list[i]
            gen_canonical = canonicalize_smiles(
                gen_smiles,
                consider_stereochemistry=consider_stereochemistry,
            )
            tgt_canonical = canonicalize_smiles(
                tgt_smiles,
                consider_stereochemistry=consider_stereochemistry,
            )
            if gen_canonical is not None:
                valid_generated += 1

            is_correct = False
            if gen_canonical is not None and tgt_canonical is not None:
                if gen_canonical == tgt_canonical:
                    canonical_correct += 1
                    is_correct = True

            # Compute Tanimoto similarity
            tanimoto = compute_tanimoto_similarity(gen_smiles, tgt_smiles)
            if tanimoto is not None:
                tanimoto_scores.append(tanimoto)

            results.append({
                'target': tgt_smiles,
                'generated': gen_smiles,
                'target_canonical': tgt_canonical,
                'generated_canonical': gen_canonical,
                'is_correct': is_correct,
                'tanimoto': tanimoto,
            })

        # Update progress bar
        if total_samples > 0:
            current_acc = canonical_correct / total_samples
            current_valid = valid_generated / total_samples
            current_tanimoto = sum(tanimoto_scores) / len(tanimoto_scores) if tanimoto_scores else 0
            pbar.set_postfix({
                "Acc": f"{current_acc:.4f}",
                "Valid": f"{current_valid:.4f}",
                "Tanimoto": f"{current_tanimoto:.4f}",
            })

    canonical_acc = canonical_correct / total_samples if total_samples > 0 else 0
    valid_rate = valid_generated / total_samples if total_samples > 0 else 0
    avg_tanimoto = sum(tanimoto_scores) / len(tanimoto_scores) if tanimoto_scores else 0

    return {
        'canonical_acc': canonical_acc,
        'valid_rate': valid_rate,
        'avg_tanimoto': avg_tanimoto,
        'total_samples': total_samples,
        'canonical_correct': canonical_correct,
        'valid_generated': valid_generated,
        'tanimoto_count': len(tanimoto_scores),
        'results': results,
    }


@torch.no_grad()
def run_beam_search_inference(
    model, dataloader, tokenizer, device,
    beam_size=10, max_gen_len=256, length_penalty=1.0,
    use_formula_embedding=False, top_k_list=None,
    consider_stereochemistry=False,
):
    """
    Run beam search inference and compute top-k accuracies.

    Args:
        beam_size: Number of beams (should be >= max(top_k_list))
        length_penalty: Length penalty for beam search scoring
        top_k_list: List of k values for computing top-k accuracy (e.g., [1, 5, 10])
                   If None, defaults to [1, 5, 10]

    Returns:
        Dictionary with top-k accuracies and detailed results
    """
    if top_k_list is None:
        top_k_list = [1, 5, 10]

    # Validate top_k_list
    top_k_list = sorted(top_k_list)
    if beam_size < max(top_k_list):
        raise ValueError(f"beam_size ({beam_size}) must be >= max(top_k_list) ({max(top_k_list)})")

    bos_idx = tokenizer.bos_token_id
    eos_idx = tokenizer.eos_token_id

    total_samples = 0
    # Dynamic counters for each k value
    top_k_correct = {k: 0 for k in top_k_list}
    valid_generated = 0
    # Store max Tanimoto scores for each top-k range
    top_k_max_tanimoto = {k: [] for k in top_k_list}
    results = []

    pbar = tqdm(dataloader, desc="Beam Search Inference")

    for batch in pbar:
        shifts = batch["raw_shifts"].to(device)
        counts = batch["counts"].to(device)
        types = batch["types"].to(device)
        nmr_padding_mask = batch["nmr_padding_mask"].to(device)
        smiles_list = batch["smiles_list"]
        formula_counts = batch["formula_counts"].to(device) if use_formula_embedding else None

        batch_size = shifts.shape[0]

        # Beam search generation
        # sequences: (B, beam_size, max_len), scores: (B, beam_size)
        sequences, scores = model.beam_search_generate(
            shifts=shifts,
            counts=counts,
            types=types,
            nmr_padding_mask=nmr_padding_mask,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            formula_counts=formula_counts,
            beam_size=beam_size,
            max_len=max_gen_len,
            length_penalty=length_penalty,
        )

        for i in range(batch_size):
            total_samples += 1
            tgt_smiles = smiles_list[i]
            tgt_canonical = canonicalize_smiles(
                tgt_smiles,
                consider_stereochemistry=consider_stereochemistry,
            )

            # Decode all beam candidates for this sample
            beam_candidates = []
            for beam_idx in range(beam_size):
                gen_smiles = decode_smiles(sequences[i, beam_idx, 1:], tokenizer, eos_idx)
                gen_canonical = canonicalize_smiles(
                    gen_smiles,
                    consider_stereochemistry=consider_stereochemistry,
                )
                # Compute Tanimoto for each candidate
                tanimoto = compute_tanimoto_similarity(gen_smiles, tgt_smiles)
                beam_candidates.append({
                    'smiles': gen_smiles,
                    'canonical': gen_canonical,
                    'score': scores[i, beam_idx].item(),
                    'tanimoto': tanimoto,
                })

            # Compute max Tanimoto for each top-k range
            for k in top_k_list:
                top_k_tanimotos = [c['tanimoto'] for c in beam_candidates[:k] if c['tanimoto'] is not None]
                if top_k_tanimotos:
                    top_k_max_tanimoto[k].append(max(top_k_tanimotos))

            # Find the rank of the correct answer (if exists)
            correct_rank = None
            for rank, candidate in enumerate(beam_candidates):
                gen_canonical = candidate['canonical']

                if gen_canonical is not None:
                    if rank == 0:
                        valid_generated += 1

                    if tgt_canonical is not None and gen_canonical == tgt_canonical:
                        correct_rank = rank
                        break  # Found match, no need to check further

            # Update counters for each k value
            top_k_status = {}
            for k in top_k_list:
                is_correct = correct_rank is not None and correct_rank < k
                top_k_status[f'is_top{k}'] = is_correct
                if is_correct:
                    top_k_correct[k] += 1

            results.append({
                'target': tgt_smiles,
                'target_canonical': tgt_canonical,
                'beam_candidates': beam_candidates,
                'correct_rank': correct_rank,
                **top_k_status,
            })

        # Update progress bar
        if total_samples > 0:
            postfix = {}
            for k in top_k_list:
                acc = top_k_correct[k] / total_samples
                postfix[f"Top{k}"] = f"{acc:.4f}"
            # Add max Tanimoto for the largest k to progress bar
            max_k = max(top_k_list)
            if top_k_max_tanimoto[max_k]:
                avg_tan = sum(top_k_max_tanimoto[max_k]) / len(top_k_max_tanimoto[max_k])
                postfix[f"MaxTan{max_k}"] = f"{avg_tan:.4f}"
            pbar.set_postfix(postfix)

    # Compute final accuracies and max Tanimoto for each k
    top_k_acc = {}
    top_k_tanimoto = {}
    for k in top_k_list:
        top_k_acc[f'top{k}_acc'] = top_k_correct[k] / total_samples if total_samples > 0 else 0
        top_k_acc[f'top{k}_correct'] = top_k_correct[k]
        # Compute avg max Tanimoto for this k
        if top_k_max_tanimoto[k]:
            top_k_tanimoto[f'top{k}_max_tanimoto'] = sum(top_k_max_tanimoto[k]) / len(top_k_max_tanimoto[k])
            top_k_tanimoto[f'top{k}_tanimoto_count'] = len(top_k_max_tanimoto[k])
        else:
            top_k_tanimoto[f'top{k}_max_tanimoto'] = 0
            top_k_tanimoto[f'top{k}_tanimoto_count'] = 0

    valid_rate = valid_generated / total_samples if total_samples > 0 else 0

    return {
        **top_k_acc,
        **top_k_tanimoto,
        'valid_rate': valid_rate,
        'total_samples': total_samples,
        'valid_generated': valid_generated,
        'top_k_list': top_k_list,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="NMR2SMILES Inference on NMRGym")
    parser.add_argument("--checkpoint", type=str, default='model_checkpoint/checkpoints_nmrgym_formula/model_best.pth')
    parser.add_argument("--test_data", type=str, default='data/case.jsonl')
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--output", type=str, default='./denovo_results/case.json')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--consider_stereochemistry",
        action="store_true",
        help="Use stereochemistry when canonicalizing SMILES for evaluation",
    )
    # Beam search arguments
    parser.add_argument("--use_beam_search", action="store_true", help="Use beam search instead of greedy decoding")
    parser.add_argument("--beam_size", type=int, default=10, help="Beam size for beam search (default: 10)")
    parser.add_argument("--length_penalty", type=float, default=1, help="Length penalty for beam search (default: 1.0)")
    parser.add_argument("--top_k", type=str, default="1,5,10", help="Comma-separated list of k values for top-k accuracy (default: '1,5,10')")
    args = parser.parse_args()

    # Parse top_k argument
    top_k_list = [int(k.strip()) for k in args.top_k.split(',')]
    args.checkpoint = resolve_repo_path(args.checkpoint)
    args.test_data = resolve_repo_path(args.test_data)
    if args.config and not os.path.isabs(args.config):
        args.config = resolve_repo_path(args.config)
    if args.output and not os.path.isabs(args.output):
        args.output = resolve_repo_path(args.output)
    if args.output:
        output_parent = os.path.dirname(args.output)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        config_base = Path(args.config).resolve().parent
    else:
        config = {
            "tokenizer_type": "chemberta", 
            "tokenizer_path": str(REPO_ROOT / "Chemberta_ckpt"),
            "d_model": 768,
            "nhead": 12,
            "num_encoder_layers": 16,
            "dim_feedforward": 3072,
            "dropout": 0.1,
            "num_decoder_layers": 6,
            "decoder_nhead": 12,
            "decoder_dim_feedforward": 3072,
            "h_bin_size": 0.01,
            "h_max": 16.0,
            "c_bin_size": 0.1,
            "c_max": 230.0,
            "use_identity_embedding": True,
            "use_count_embedding": False,
            "max_smiles_len": 256,
            "use_count_fusion": False,
            "use_formula_embedding": True,
            "num_atom_types": 50,
        }
        config_base = REPO_ROOT

    tokenizer_path = config.get("tokenizer_path")
    if isinstance(tokenizer_path, str) and not os.path.isabs(tokenizer_path):
        config["tokenizer_path"] = str((config_base / tokenizer_path).resolve())

    # Auto-detect formula embedding from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu',weights_only=True)
    has_formula_embedding = any('formula_embedding' in k for k in checkpoint['model_state_dict'].keys())
    config['use_formula_embedding'] = has_formula_embedding
    print(f"Use formula embedding: {has_formula_embedding}")

    # Load model
    model, tokenizer = load_model(args.checkpoint, config, device)

    # Load test data
    print(f"Loading test data: {args.test_data}")
    test_dataset = NMRGymTestDataset(args.test_data, tokenizer, config.get('max_smiles_len', 256))
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(nmrgym_collate_fn, pad_idx=tokenizer.pad_token_id),
        num_workers=4,
        pin_memory=True,
    )

    # Run inference
    if args.use_beam_search:
        print(f"\n{'=' * 60}")
        print(f"Running beam search inference:")
        print(f"  Beam size: {args.beam_size}")
        print(f"  Length penalty: {args.length_penalty}")
        print(f"  Top-k values: {top_k_list}")
        print(f"{'=' * 60}\n")

        results = run_beam_search_inference(
            model, test_loader, tokenizer, device,
            beam_size=args.beam_size,
            max_gen_len=args.max_gen_len,
            length_penalty=args.length_penalty,
            use_formula_embedding=config['use_formula_embedding'],
            top_k_list=top_k_list,
            consider_stereochemistry=args.consider_stereochemistry,
        )

        # Print results
        print("\n" + "=" * 60)
        print("Beam Search Results:")
        print(f"  Total samples:     {results['total_samples']}")
        print(f"  Valid SMILES rate: {results['valid_rate']:.4f} ({results['valid_generated']}/{results['total_samples']})")
        print("-" * 60)
        print(f"  {'Top-k':<8} {'Accuracy':<12} {'Max Tanimoto':<12}")
        print("-" * 60)
        for k in results['top_k_list']:
            acc = results[f'top{k}_acc']
            acc_count = results[f'top{k}_correct']
            max_tan = results[f'top{k}_max_tanimoto']
            print(f"  Top-{k:<4} {acc:.4f} ({acc_count:>4}/{results['total_samples']})  {max_tan:.4f}")
        print("=" * 60)

        # Save results
        if args.output:
            output_dict = {
                'valid_rate': results['valid_rate'],
                'total_samples': results['total_samples'],
                'beam_size': args.beam_size,
                'length_penalty': args.length_penalty,
                'top_k_list': results['top_k_list'],
                'predictions': results['results'],
            }
            # Add top-k accuracy and max tanimoto results
            for k in results['top_k_list']:
                output_dict[f'top{k}_acc'] = results[f'top{k}_acc']
                output_dict[f'top{k}_correct'] = results[f'top{k}_correct']
                output_dict[f'top{k}_max_tanimoto'] = results[f'top{k}_max_tanimoto']

            with open(args.output, 'w') as f:
                json.dump(output_dict, f, indent=2)
            print(f"Results saved to {args.output}")

        # Print examples
        print("\nSample predictions (showing top-3 candidates):")
        for i, r in enumerate(results['results'][:5]):
            top_k_status = [f"Top-{k}: {r[f'is_top{k}']}" for k in results['top_k_list']]
            print(f"\n[{i+1}] {', '.join(top_k_status)}")
            print(f"  Target: {r['target']}")
            if r['correct_rank'] is not None:
                print(f"  Correct rank: {r['correct_rank'] + 1}")
            else:
                print(f"  Correct rank: Not found in top-{args.beam_size}")
            print(f"  Candidates:")
            for j, cand in enumerate(r['beam_candidates'][:3]):
                tan_str = f"{cand['tanimoto']:.4f}" if cand['tanimoto'] is not None else "N/A"
                print(f"    [{j+1}] {cand['smiles']} (score: {cand['score']:.4f}, tanimoto: {tan_str})")

    else:
        results = run_inference(
            model, test_loader, tokenizer, device,
            max_gen_len=args.max_gen_len,
            use_formula_embedding=config['use_formula_embedding'],
            consider_stereochemistry=args.consider_stereochemistry,
        )

        # Print results
        print("\n" + "=" * 60)
        print("Results:")
        print(f"  Total samples:     {results['total_samples']}")
        print(f"  Valid SMILES rate: {results['valid_rate']:.4f} ({results['valid_generated']}/{results['total_samples']})")
        print(f"  Canonical accuracy:{results['canonical_acc']:.4f} ({results['canonical_correct']}/{results['total_samples']})")
        print(f"  Avg Tanimoto:      {results['avg_tanimoto']:.4f} ({results['tanimoto_count']}/{results['total_samples']})")
        print("=" * 60)

        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'canonical_acc': results['canonical_acc'],
                    'valid_rate': results['valid_rate'],
                    'avg_tanimoto': results['avg_tanimoto'],
                    'tanimoto_count': results['tanimoto_count'],
                    'total_samples': results['total_samples'],
                    'predictions': results['results'],
                }, f, indent=2)
            print(f"Results saved to {args.output}")

        # Print examples
        print("\nSample predictions:")
        for i, r in enumerate(results['results'][:5]):
            tanimoto_str = f"{r['tanimoto']:.4f}" if r['tanimoto'] is not None else "N/A"
            print(f"\n[{i+1}] Correct: {r['is_correct']}, Tanimoto: {tanimoto_str}")
            print(f"  Target:    {r['target']}")
            print(f"  Generated: {r['generated']}")


if __name__ == "__main__":
    main()
