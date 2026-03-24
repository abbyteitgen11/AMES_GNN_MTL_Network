
""" defines the device; to be imported by all modules """

import os

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
   num_workers = 0 # os.cpu_count()
else:
   num_workders = 0
