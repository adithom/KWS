import sys
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import subprocess
import glob

def findCUDA():
    '''Finds the CUDA install path.'''
    IS_WINDOWS = sys.platform == 'win32'
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    
    if cuda_home is None:
        # Try locating nvcc binary using `which` command for Unix-based systems
        try:
            which = 'where' if IS_WINDOWS else 'which'
            nvcc = subprocess.check_output([which, 'nvcc']).decode().strip()
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            pass

    # Check typical installation paths if CUDA is still not found
    if cuda_home is None:
        if IS_WINDOWS:
            cuda_homes = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
            cuda_home = cuda_homes[0] if cuda_homes else None
        else:
            # Typical CUDA paths on Unix-based systems
            common_paths = ['/usr/local/cuda', '/usr/local/cuda-11', '/usr/local/cuda-12']
            cuda_home = next((path for path in common_paths if os.path.exists(path)), None)
            
    return cuda_home

# Example usage
cuda_path = findCUDA()
print(f"CUDA path: {cuda_path if cuda_path else 'Not found'}")
