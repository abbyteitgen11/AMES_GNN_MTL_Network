import numpy as np
import os
import random
import torch
"""
Set random seeds for consistency

"""

def set_seed(s):
    seed_value = s

    # Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # Set the `pytorch` pseudo-random generator at a fixed value
    torch.manual_seed(seed_value)

    # If using CUDA, set the seed for all GPUs
    #torch.cuda.manual_seed(seed_value)
    #torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups

    # Ensure deterministic behavior (slower but fully reproducible)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
