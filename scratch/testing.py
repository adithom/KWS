from data_pipeline.mfccProcessor import MFCCProcessor
import matplotlib.pyplot as plt

# def findCUDA():
#     '''Finds the CUDA install path.'''
#     IS_WINDOWS = sys.platform == 'win32'
#     cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
#
#     if cuda_home is None:
#         # Try locating nvcc binary using `which` command for Unix-based systems
#         try:
#             which = 'where' if IS_WINDOWS else 'which'
#             nvcc = subprocess.check_output([which, 'nvcc']).decode().strip()
#             cuda_home = os.path.dirname(os.path.dirname(nvcc))
#         except Exception:
#             pass
#
#     # Check typical installation paths if CUDA is still not found
#     if cuda_home is None:
#         if IS_WINDOWS:
#             cuda_homes = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
#             cuda_home = cuda_homes[0] if cuda_homes else None
#         else:
#             # Typical CUDA paths on Unix-based systems
#             common_paths = ['/usr/local/cuda', '/usr/local/cuda-11', '/usr/local/cuda-12']
#             cuda_home = next((path for path in common_paths if os.path.exists(path)), None)
#
#     return cuda_home
#
# print(findCUDA())
# print(torch.__version__)
# print(torch.version.cuda)

processor = MFCCProcessor()
features = processor.compute_features('../google_speech/bed/0a7c2a8d_nohash_0.wav')

if features is not None:
    print(f"MFCC shape: {features.shape}")

    # Render the MFCC as a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(features, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title('MFCC Features')
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time Frames')
    plt.show()
else:
    print("MFCC computation failed.")

