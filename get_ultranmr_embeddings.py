import argparse
import json
from pathlib import Path

import torch

from models.UltraNMR import UltraNMR


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CASE_PATH = REPO_ROOT / "data" / "case.jsonl"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "model_checkpoint" / "checkpoints_nce" / "model_epoch_1.pth"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def load_cases(case_path: Path):
    cases = []
    with case_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {case_path}:{line_no}") from exc

            missing = {"smiles", "h_shift", "c_shift"} - item.keys()
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(f"Missing keys [{missing_str}] at {case_path}:{line_no}")

            cases.append(item)
    return cases


def build_batch(case):
    h_shifts = torch.tensor(case["h_shift"], dtype=torch.float32)
    c_shifts = torch.tensor(case["c_shift"], dtype=torch.float32)

    all_shifts = torch.cat([h_shifts, c_shifts], dim=0)
    counts = torch.cat([
        torch.ones_like(h_shifts),
        torch.zeros_like(c_shifts),
    ], dim=0)
    types = torch.cat([
        torch.zeros_like(h_shifts, dtype=torch.long),
        torch.ones_like(c_shifts, dtype=torch.long),
    ], dim=0)
    padding_mask = torch.zeros_like(all_shifts, dtype=torch.bool)

    return {
        "shifts": all_shifts.unsqueeze(0),
        "counts": counts.unsqueeze(0),
        "types": types.unsqueeze(0),
        "padding_mask": padding_mask.unsqueeze(0),
    }


def mean_pool(sequence_output, padding_mask):
    valid_mask = (~padding_mask).unsqueeze(-1).float()
    pooled = (sequence_output * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)
    return pooled


def load_model(checkpoint_path: Path, device: torch.device):
    model = UltraNMR(
        d_model=768,
        nhead=12,
        num_encoder_layers=16,
        dim_feedforward=3072,
        dropout=0.1,
        h_bin_size=0.01,
        h_max=16.0,
        c_bin_size=0.1,
        c_max=230.0,
        fp_sim_bin_size=0.005,
        fp_sim_mlp_hidden=512,
        use_identity_embedding=True,
        use_count_embedding=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Get UltraNMR embeddings for cases in JSONL")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--input", type=str, default=str(DEFAULT_CASE_PATH))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--print_values", action="store_true", help="Print full embedding values")
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    case_path = resolve_path(args.input)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cases = load_cases(case_path)
    model = load_model(checkpoint_path, device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input: {case_path}")
    print(f"Device: {device}")
    print(f"Cases: {len(cases)}")

    with torch.no_grad():
        for idx, case in enumerate(cases, start=1):
            batch = build_batch(case)
            shifts = batch["shifts"].to(device)
            counts = batch["counts"].to(device)
            types = batch["types"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            _, _, sequence_output = model(shifts, counts, types, padding_mask)
            pooled_embedding = mean_pool(sequence_output, padding_mask)

            print(f"\nCase {idx}")
            print(f"SMILES: {case['smiles']}")
            print(f"Num H peaks: {len(case['h_shift'])}")
            print(f"Num C peaks: {len(case['c_shift'])}")
            print(f"Sequence embedding shape: {tuple(sequence_output.shape)}")
            print(f"Mean pooled embedding shape: {tuple(pooled_embedding.shape)}")

            if args.print_values:
                print("Sequence embedding:")
                print(sequence_output.squeeze(0).cpu())
                print("Mean pooled embedding:")
                print(pooled_embedding.squeeze(0).cpu())


if __name__ == "__main__":
    main()
