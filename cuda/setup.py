import setuptools #enables develop
import os
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import findCUDA

if findCUDA() is not None:
    setuptools.setup(
        name='fastgrnn_cuda',
        ext_modules=[
            CUDAExtension('fastgrnn_cuda', [
                'fastgrnn_cuda.cpp',
                'fastgrnn_cuda_kernel.cu',
            ]),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )