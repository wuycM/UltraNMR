import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import lmdb
import pickle
import random  # Used to generate shift perturbations.

class NMRDataset(Dataset):
    def __init__(self, data_source,
                 h_x_min=0.01, h_x_max=16.0,
                 c_x_min=0.01, c_x_max=230.0,
                 lmdb_max_readers=256,
                 h_shift_range=(-0.01, 0.01),  # H-spectrum perturbation range
                 c_shift_range=(-0.1, 0.1),  # C-spectrum perturbation range
                 shift_aug_p=0.2):  # Probability of applying shift augmentation
        """
        data_source: Path to the LMDB file.
        h_x_min/x_max, c_x_min/x_max: H/C range parameters, kept for compatibility
            but no longer used for normalization.
        lmdb_max_readers: Maximum number of LMDB readers.
        h_shift_range: Perturbation range for H spectra.
        c_shift_range: Perturbation range for C spectra.
        shift_aug_p: Probability of applying shift augmentation.
        """
        self.h_x_min = h_x_min
        self.h_x_max = h_x_max
        self.c_x_min = c_x_min
        self.c_x_max = c_x_max
        self.h_shift_range = h_shift_range  # H-spectrum perturbation range
        self.c_shift_range = c_shift_range  # C-spectrum perturbation range
        self.shift_aug_p = shift_aug_p  # Probability of applying shift augmentation
        
        self.env = lmdb.open(
            data_source,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=lmdb_max_readers,
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        # with self.env.begin(write=False) as txn:
        #     self.length = min(txn.stat()['entries'], 10000)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            value = txn.get(str(idx).encode())
            if value is None:
                raise IndexError(f"Index {idx} not found in LMDB")
            sample = pickle.loads(value)

        # -------------------- Extract raw spectral data --------------------
        c_shifts = torch.tensor(sample.get('C_nmr', []), dtype=torch.float32).unique()
        h_info = sample.get('H_nmr', [])
        h_shifts = torch.tensor([item['shift'] for item in h_info], dtype=torch.float32)
        h_counts = torch.tensor([item['count'] for item in h_info], dtype=torch.float32)

        # Extract the C shift corresponding to each H peak from the c_shift field.
        h_c_shifts = torch.tensor([item.get('c_shift', 0.0) for item in h_info], dtype=torch.float32)
        # -------------------- Shift perturbation --------------------
        # Randomly decide whether to apply perturbation.
        if random.random() < self.shift_aug_p:
            # Perturb the H spectrum.
            h_shift_bias = random.uniform(self.h_shift_range[0], self.h_shift_range[1])
            h_shifts += h_shift_bias

            # Perturb the C spectrum.
            c_shift_bias = random.uniform(self.c_shift_range[0], self.c_shift_range[1])
            c_shifts += c_shift_bias

        all_shifts = torch.cat([h_shifts, c_shifts])  # Raw ppm values, used directly without normalization
        c_counts = torch.zeros_like(c_shifts)
        all_counts = torch.cat([h_counts, c_counts])
        types = torch.cat([
            torch.zeros_like(h_shifts, dtype=torch.long),
            torch.ones_like(c_shifts, dtype=torch.long)
        ])

        # -------------------- Extract SMILES and compute Morgan fingerprint --------------------
        smiles = sample.get('smiles', '')

        # If fp is already stored in LMDB in binary form, use it directly.
        if 'fp' in sample and sample['fp'] is not None:
            fp = np.frombuffer(sample['fp'], dtype=np.uint8)
            fp = np.unpackbits(fp)  # Decode into binary bits
            fingerprint = torch.tensor(fp, dtype=torch.float32)
        else:
            # Otherwise compute the Morgan fingerprint from SMILES.
            mol = Chem.MolFromSmiles(smiles)
            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            if mol is not None:
                fp = morgan_gen.GetFingerprint(mol)
                fingerprint = torch.tensor(np.array(fp), dtype=torch.float32)
            else:
                # If the SMILES is invalid, use a zero vector.
                fingerprint = torch.zeros(2048, dtype=torch.float32)

        # No normalization is applied anymore; raw_shifts and shifts use the same raw ppm values.
        return {
            "raw_shifts": all_shifts,  # For FourierFeatures (raw ppm)
            "shifts": all_shifts,      # For loss / regression (raw ppm)
            "counts": all_counts,
            "types": types,
            "fingerprint": fingerprint,  # Morgan fingerprint
            "smiles": smiles,  # Preserve SMILES for debugging
            "h_c_shifts": h_c_shifts,  # C shift corresponding to each H peak
            "c_shifts": c_shifts,  # All C shifts, used as negatives for InfoNCE
        }


def collate_fn(batch):
    # Pad each field separately.
    raw_shifts = [item['raw_shifts'] for item in batch]
    shifts = [item['shifts'] for item in batch]
    counts = [item['counts'] for item in batch]
    types = [item['types'] for item in batch]
    fingerprints = [item['fingerprint'] for item in batch]  # Morgan fingerprints
    h_c_shifts = [item['h_c_shifts'] for item in batch]  # C shift corresponding to each H peak
    c_shifts_list = [item['c_shifts'] for item in batch]  # All C shifts for each sample

    raw_shifts_padded = pad_sequence(raw_shifts, batch_first=True, padding_value=0.0)
    shifts_padded = pad_sequence(shifts, batch_first=True, padding_value=0.0)
    counts_padded = pad_sequence(counts, batch_first=True, padding_value=0.0)
    types_padded = pad_sequence(types, batch_first=True, padding_value=-1)

    # Pad h_c_shifts, which stores the C shift for each H peak.
    h_c_shifts_padded = pad_sequence(h_c_shifts, batch_first=True, padding_value=0.0)

    padding_mask = (types_padded == -1)
    types_padded[padding_mask] = 0  # Fix the padding placeholder.

    # Stack fingerprints (all have same size)
    fingerprints_stacked = torch.stack(fingerprints, dim=0)  # [B, 2048]

    return {
        "raw_shifts": raw_shifts_padded,  # For FourierFeatures
        "shifts": shifts_padded,          # For loss
        "counts": counts_padded,
        "types": types_padded,
        "padding_mask": padding_mask,
        "fingerprints": fingerprints_stacked,  # [B, 2048]
        "h_c_shifts": h_c_shifts_padded,  # [B, num_H] C shift corresponding to each H peak
        "c_shifts_list": c_shifts_list,  # List[Tensor] all C shifts per sample, kept unpadded for InfoNCE
    }
