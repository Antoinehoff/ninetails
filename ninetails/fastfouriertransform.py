import numpy as np
import warnings

try:
    import cupy as cp
    cp_available = True
except ImportError:
    warnings.warn("CuPy not found. GPU FFTs will not be available.")
    cp_available = False

class FastFourierTransform:
    def __init__(self, norm='ortho', axes=(-2, -1), size=None):
        self.norm = norm
        self.axes = axes
        self.size = size
        self.gpu_available = cp_available and self._check_gpu()
        if self.gpu_available:
            self.rfft2 = self.rfft2_gpu
            self.irfft2 = self.irfft2_gpu
            self.fft2 = self.fft2_gpu
            self.ifft2 = self.ifft2_gpu
        else:
            self.rfft2 = self.rfft2_cpu
            self.irfft2 = self.irfft2_cpu
            self.fft2 = self.fft2_cpu
            self.ifft2 = self.ifft2_cpu
        
    def _check_gpu(self):
        try:
            cp.cuda.Device(0)
            return True
        except:
            warnings.warn("No CUDA-capable GPU detected.")
            return False
    
    def fft2_cpu(self, x):
        return np.fft.fft2(x, s=self.size, axes=self.axes, norm=self.norm)
    
    def ifft2_cpu(self, y):
        return np.fft.ifft2(y, s=self.size, axes=self.axes, norm=self.norm).real
    
    def fft2_gpu(self, x):
        x_gpu = cp.asarray(x, dtype=cp.float32)
        y_gpu = cp.fft.rfft2(x_gpu, s=self.size, axes=self.axes, norm=self.norm)
        return cp.asnumpy(y_gpu)
    
    def ifft2_gpu(self, y):
        y_gpu = cp.asarray(y, dtype=cp.complex64)
        x_gpu = cp.fft.irfft2(y_gpu, s=self.size, axes=self.axes, norm=self.norm)
        return cp.asnumpy(x_gpu)
    
    def rfft2_cpu(self, x):
        return np.fft.rfft2(x, s=self.size, axes=self.axes, norm=self.norm)
    
    def irfft2_cpu(self, y):
        return np.fft.irfft2(y, s=self.size, axes=self.axes, norm=self.norm)
    
    def rfft2_gpu(self, x):
        x_gpu = cp.asarray(x, dtype=cp.float32)
        y_gpu = cp.fft.rfft2(x_gpu, s=self.size, axes=self.axes, norm=self.norm)  
        return cp.asnumpy(y_gpu)
    
    def irfft2_gpu(self, y):
        y_gpu = cp.asarray(y, dtype=cp.complex64)
        x_gpu = cp.fft.irfft2(y_gpu, s=self.size, axes=self.axes, norm=self.norm)
        return cp.asnumpy(x_gpu)