import cupy as cp

try:
    # Allocate a small array on the GPU
    x_gpu = cp.array([1], dtype=cp.float32)

    # Perform a simple computation to ensure CUDA is working
    x_gpu += 1

    print("CUDA works with CuPy!")
except cp.cuda.runtime.CUDARuntimeError as e:
    print("CUDA is not available:", e)