import numpy as np
import cupy as cp  # Use CuPy for GPU FFT
import warnings

class FastFourierTransform:
    def __init__(self):
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self):
        """Check if a CUDA-capable GPU is available."""
        try:
            cp.cuda.Device(0)  # Check if CuPy detects a GPU
            return True
        except:
            warnings.warn("No CUDA-capable GPU detected.")
            return False
    
    def fft2_cpu(self, x, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Compute 2D FFT on CPU."""
        y = np.fft.fft2(x, s=s, axes=axes, norm=norm)
        return np.fft.fftshift(y) if fftshift else y
    
    def ifft2_cpu(self, y, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Compute 2D inverse FFT on CPU."""
        if fftshift:
            y = np.fft.ifftshift(y)
        return np.fft.ifft2(y, s=s, axes=axes, norm=norm).real
    
    def fft2_gpu(self, x, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Compute 2D FFT on GPU using CuPy."""
        if not self.gpu_available:
            raise RuntimeError("No CUDA-capable GPU detected.")
        
        x_gpu = cp.asarray(x, dtype=cp.float32)
        y_gpu = cp.fft.rfft2(x_gpu, s=s, axes=axes, norm=norm)
        
        return cp.asnumpy(cp.fft.fftshift(y_gpu) if fftshift else y_gpu)
    
    def ifft2_gpu(self, y, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Compute 2D inverse FFT on GPU using CuPy."""
        if not self.gpu_available:
            raise RuntimeError("No CUDA-capable GPU detected.")
        
        y_gpu = cp.asarray(y, dtype=cp.complex64)
        if fftshift:
            y_gpu = cp.fft.ifftshift(y_gpu)
        x_gpu = cp.fft.irfft2(y_gpu, s=s, axes=axes, norm=norm)
        
        return cp.asnumpy(x_gpu)
    
    def fft2(self, x, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Auto-selects GPU if available, otherwise runs CPU FFT."""
        return self.fft2_gpu(x, fftshift, norm, axes, s) if self.gpu_available else self.fft2_cpu(x, fftshift, norm, axes, s)
    
    def ifft2(self, y, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Auto-selects GPU if available, otherwise runs CPU inverse FFT."""
        return self.ifft2_gpu(y, fftshift, norm, axes, s) if self.gpu_available else self.ifft2_cpu(y, fftshift, norm, axes, s)
    
    def rfft2_cpu(self, x, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Compute 2D real FFT on CPU."""
        y = np.fft.rfft2(x, s=s, axes=axes, norm=norm)
        return np.fft.fftshift(y) if fftshift else y
    
    def irfft2_cpu(self, y, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Compute 2D inverse real FFT on CPU."""
        if fftshift:
            y = np.fft.ifftshift(y)
        return np.fft.irfft2(y, s=s, axes=axes, norm=norm)
    
    def rfft2_gpu(self, x, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Compute 2D real FFT on GPU using CuPy."""
        if not self.gpu_available:
            raise RuntimeError("No CUDA-capable GPU detected.")
        
        x_gpu = cp.asarray(x, dtype=cp.float32)
        y_gpu = cp.fft.rfft2(x_gpu, s=s, axes=axes, norm=norm)
        
        return cp.asnumpy(cp.fft.fftshift(y_gpu) if fftshift else y_gpu)
    
    def irfft2_gpu(self, y, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Compute 2D inverse real FFT on GPU using CuPy."""
        if not self.gpu_available:
            raise RuntimeError("No CUDA-capable GPU detected.")
        
        y_gpu = cp.asarray(y, dtype=cp.complex64)
        if fftshift:
            y_gpu = cp.fft.ifftshift(y_gpu)
        x_gpu = cp.fft.irfft2(y_gpu, s=s, axes=axes, norm=norm)
        
        return cp.asnumpy(x_gpu)
    
    def rfft2(self, x, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Auto-selects GPU if available, otherwise runs CPU real FFT."""
        return self.rfft2_gpu(x, fftshift, norm, axes, s) if self.gpu_available else self.rfft2_cpu(x, fftshift, norm, axes, s)
    
    def irfft2(self, y, fftshift=False, norm=None, axes=(-2, -1), s=None):
        """Auto-selects GPU if available, otherwise runs CPU inverse real FFT."""
        return self.irfft2_gpu(y, fftshift, norm, axes, s) if self.gpu_available else self.irfft2_cpu(y, fftshift, norm, axes, s)