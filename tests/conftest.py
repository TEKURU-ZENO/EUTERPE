"""Global PyTest configuration and pre-import hooks."""

# Pre-import FAISS and PyTorch to safely bind native DLL handles on Windows before pytest assertion rewriting
try:
    import faiss
except Exception:
    pass

import torch
import numpy as np
