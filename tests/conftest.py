import os

# This project uses PyTorch only; avoid importing unrelated inherited TF/JAX stacks.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["OMP_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)
