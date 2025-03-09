import numpy as np
from ninetails import FastFourierTransform
import cupy as cp

gpu_available = True

# Initialize the FFT handler
fft_handler = FastFourierTransform()

# Define dimensions
dim_x, dim_y = 256, 256

# Create a sample 2D array (e.g., a simple image or matrix)
test_array = np.random.rand(dim_x, dim_y).astype(np.float32)

# Compute FFT using CPU
print("Running CPU FFT...")
fft_cpu = fft_handler.fft2_cpu(test_array, fftshift=True, norm='ortho', s=(dim_x, dim_y))
print("Running CPU IFFT...")
ifft_cpu = fft_handler.ifft2_cpu(fft_cpu, fftshift=True, norm='ortho', s=(dim_x, dim_y))

# Compute real FFT using CPU
print("Running CPU real FFT...")
rfft_cpu = fft_handler.rfft2_cpu(test_array, fftshift=True, norm='ortho', s=(dim_x, dim_y))
print("Running CPU real IFFT...")
irfft_cpu = fft_handler.irfft2_cpu(rfft_cpu, fftshift=True, norm='ortho', s=(dim_x, dim_y))

# Check if GPU is available before running GPU FFT
if gpu_available:
    cp._default_memory_pool.free_all_blocks()  # Free up memory before running GPU FFT
    print("Running GPU FFT...")
    fft_gpu = fft_handler.fft2_gpu(test_array, fftshift=True, norm='ortho', s=(dim_x, dim_y))
    print("Running GPU IFFT...")
    ifft_gpu = fft_handler.ifft2_gpu(fft_gpu, fftshift=True, norm='ortho', s=(dim_x, dim_y))
    
    cp._default_memory_pool.free_all_blocks()  # Free up memory before running GPU FFT
    print("Running GPU real FFT...")
    rfft_gpu = fft_handler.rfft2_gpu(test_array, fftshift=True, norm='ortho', s=(dim_x, dim_y))
    print("Running GPU real IFFT...")
    irfft_gpu = fft_handler.irfft2_gpu(rfft_gpu, fftshift=True, norm='ortho', s=(dim_x, dim_y))
else:
    print("GPU not available. Only CPU FFT was tested.")
    print("GPU not available. Only CPU real FFT was tested.")

# Validate the result
print("Max difference between original and IFFT (CPU):", np.max(np.abs(test_array - ifft_cpu)))
if gpu_available:
    print("Max difference between original and IFFT (GPU):", np.max(np.abs(test_array - ifft_gpu)))

# Validate the result for real FFT
print("Max difference between original and real IFFT (CPU):", np.max(np.abs(test_array - irfft_cpu)))
if gpu_available:
    print("Max difference between original and real IFFT (GPU):", np.max(np.abs(test_array - irfft_gpu)))