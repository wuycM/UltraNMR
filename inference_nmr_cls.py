import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.nmr_classifier import build_nmr_classifier
from utils.nmr_classification_dataset import (
    NMRClassificationDataset,
    nmr_classification_collate_fn,
    load_class_mapping,
)

REPO_ROOT = Path(__file__).resolve().parent


def resolve_repo_path(path_value):
    if path_value is None or os.path.isabs(path_value):
        return path_value
    return str((REPO_ROOT / path_value).resolve())

@torch.no_grad()
def predict(model, loader, device, idx_to_class, output_path):
    """Run inference and save the results to a file."""
    model.eval()
    results = []
    
    pbar = tqdm(loader, desc="Predicting")

    for batch in pbar:
        shifts = batch["raw_shifts"].to(device)
        counts = batch["counts"].to(device)
        types = batch["types"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        
        # Get the original SMILES so results can be matched back to inputs.
        smiles_list = batch.get("smiles", [""] * shifts.shape[0])

        # Forward pass
        logits = model(
            shifts=shifts,
            counts=counts,
            types=types,
            padding_mask=padding_mask
        )

        # Compute probabilities and predictions
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)

        for i in range(len(predictions)):
            pred_idx = predictions[i].item()
            res = {
                "smiles": smiles_list[i],
                "predicted_superclass": idx_to_class[pred_idx],
                "confidence": float(confidences[i].item())
            }
            results.append(res)

    # Save predictions as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nPrediction complete. Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict NMR superclass")
    parser.add_argument('--checkpoint', type=str, default='model_checkpoint/checkpoints_nmrgym_cls/classification_model_best.pth', help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config_inference_nmr_cls.json', help='Path to config JSON file')
    parser.add_argument('--input_data', type=str, default='data/NMRGym_test_balanced_with_class.jsonl', help='Path to input jsonl file')
    parser.add_argument('--output_name', type=str, default='predictions.jsonl', help='Output filename')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    args.checkpoint = resolve_repo_path(args.checkpoint)
    args.config = resolve_repo_path(args.config)
    args.input_data = resolve_repo_path(args.input_data)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Load class mapping
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_config = checkpoint.get('config', config)
    save_dir = checkpoint_config.get('save_dir', os.path.dirname(args.checkpoint))
    class_mapping_path = os.path.join(save_dir, 'class_mapping.json')
    
    class_to_idx, idx_to_class, _ = load_class_mapping(class_mapping_path)
    num_classes = len(class_to_idx)

    # Build model
    model = build_nmr_classifier(
        config=config,
        pretrained_encoder_path=None,
        num_classes=num_classes,
        head_type=checkpoint_config.get('head_type', 'linear'),
        hidden_dim=checkpoint_config.get('hidden_dim', 512),
        encoder_config=checkpoint_config,
        device='cpu'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Create dataset (set label_field to None or handle unlabeled data).
    # Make sure NMRClassificationDataset does not fail when labels are absent.
    dataset = NMRClassificationDataset(
        jsonl_path=args.input_data,
        class_to_idx=class_to_idx,
        split='test', 
        filter_unseen_classes=False # Do not filter unseen classes in prediction mode
    )

    loader = DataLoader(
        dataset,
        batch_size=config.get('val_batch_size', 256),
        shuffle=False,
        collate_fn=nmr_classification_collate_fn,
        num_workers=4
    )

    output_path = os.path.join(save_dir, args.output_name)
    predict(model, loader, device, idx_to_class, output_path)

if __name__ == '__main__':
    main()
