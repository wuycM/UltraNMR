import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Fix all random seeds to ensure reproducible experiments.

    Args:
        seed (int): Random seed value.
        deterministic (bool): Whether to enable deterministic algorithms.
            This may be slightly slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Control CuDNN randomness.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # Environment variable sometimes used by DataLoader / multiprocessing.
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed fixed to {seed} | deterministic={deterministic}")
