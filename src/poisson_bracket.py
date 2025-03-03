# poisson_bracket.py
import numpy as np

class PoissonBracket:
    def __init__(self, kx, ky):
        """
        Initialize the Poisson bracket calculator.
        
        Parameters:
        -----------
        kx : ndarray
            Radial wavenumbers (2D array)
        ky : ndarray
            Binormal wavenumbers (2D array)
        """
        self.kx = kx
        self.ky = ky
        self.nx, self.ny = kx.shape
        
        # Calculate dealiasing pad sizes (using 3/2 rule)
        self.nx_pad = int(self.nx * 3 / 2)
        self.ny_pad = int(self.ny * 3 / 2)
    
    def pad(self, f_hat):
        """
        Pad an array in Fourier space for anti-aliasing.
        
        Parameters:
        -----------
        f_hat : ndarray
            Array in Fourier space
            
        Returns:
        --------
        ndarray
            Padded array
        """
        # Get dimensions
        shape = f_hat.shape
        
        # Check if we have z dimension
        has_z = len(shape) > 2
        
        # Calculate padding offsets
        pad_x = (self.nx_pad - self.nx) // 2
        pad_y = (self.ny_pad - self.ny) // 2
        
        if has_z:
            nz = shape[2]
            f_hat_padded = np.zeros((self.nx_pad, self.ny_pad, nz), dtype=np.complex128)
            f_hat_padded[pad_x:pad_x+self.nx, pad_y:pad_y+self.ny, :] = f_hat
        else:
            f_hat_padded = np.zeros((self.nx_pad, self.ny_pad), dtype=np.complex128)
            f_hat_padded[pad_x:pad_x+self.nx, pad_y:pad_y+self.ny] = f_hat
            
        return f_hat_padded
        
    def unpad(self, f_hat_padded):
        """
        Unpad an array in Fourier space after anti-aliasing calculation.
        
        Parameters:
        -----------
        f_hat_padded : ndarray
            Padded array in Fourier space
            
        Returns:
        --------
        ndarray
            Unpadded array
        """
        # Get dimensions
        shape = f_hat_padded.shape
        
        # Check if we have z dimension
        has_z = len(shape) > 2
        
        # Calculate padding offsets
        pad_x = (self.nx_pad - self.nx) // 2
        pad_y = (self.ny_pad - self.ny) // 2
        
        if has_z:
            f_hat = f_hat_padded[pad_x:pad_x+self.nx, pad_y:pad_y+self.ny, :]
        else:
            f_hat = f_hat_padded[pad_x:pad_x+self.nx, pad_y:pad_y+self.ny]
            
        return f_hat
        
    def compute(self, a_hat, b_hat):
        """
        Compute the Poisson bracket {a,b} = (∂a/∂x)(∂b/∂y) - (∂a/∂y)(∂b/∂x)
        using a pseudo-spectral method with anti-aliasing.
        
        Parameters:
        -----------
        a_hat : ndarray
            First field in Fourier space
        b_hat : ndarray
            Second field in Fourier space
            
        Returns:
        --------
        ndarray
            Poisson bracket result in Fourier space
        """
        if a_hat.shape != b_hat.shape:
            raise ValueError("Input arrays must have the same shape")
        
        # Determine if we have a z dimension
        has_z = len(a_hat.shape) > 2
        z_dim = a_hat.shape[2] if has_z else 1
        
        # Initialize the result array
        result_shape = a_hat.shape
        result = np.zeros(result_shape, dtype=np.complex128)
        
        # Create padded wavenumbers when needed
        kx_pad_2d = np.zeros((self.nx_pad, self.ny_pad))
        ky_pad_2d = np.zeros((self.nx_pad, self.ny_pad))
        
        # Fill padded wavenumbers
        pad_x = (self.nx_pad - self.nx) // 2
        pad_y = (self.ny_pad - self.ny) // 2
        kx_pad_2d[pad_x:pad_x+self.nx, pad_y:pad_y+self.ny] = self.kx
        ky_pad_2d[pad_x:pad_x+self.nx, pad_y:pad_y+self.ny] = self.ky
        
        # Loop over z dimension if present
        for iz in range(z_dim):
            # Extract 2D slices if we have a z dimension
            a_hat_2d = a_hat[:, :, iz] if has_z else a_hat
            b_hat_2d = b_hat[:, :, iz] if has_z else b_hat
            
            # Pad the arrays for anti-aliasing
            a_hat_pad = self.pad(a_hat_2d)
            b_hat_pad = self.pad(b_hat_2d)
            
            # Compute derivatives in Fourier space
            da_dx_pad = 1j * kx_pad_2d * a_hat_pad
            da_dy_pad = 1j * ky_pad_2d * a_hat_pad
            db_dx_pad = 1j * kx_pad_2d * b_hat_pad
            db_dy_pad = 1j * ky_pad_2d * b_hat_pad
            
            # Transform to real space for multiplication
            da_dx_real = np.fft.irfft2(da_dx_pad)
            da_dy_real = np.fft.irfft2(da_dy_pad)
            db_dx_real = np.fft.irfft2(db_dx_pad)
            db_dy_real = np.fft.irfft2(db_dy_pad)
            
            # Compute Poisson bracket in real space
            pb_real = da_dx_real * db_dy_real - da_dy_real * db_dx_real
            
            # Transform back to Fourier space
            pb_hat_pad = np.fft.rfft2(pb_real)
            
            # Unpad the result
            pb_hat = self.unpad(pb_hat_pad)
            
            # Store the result
            if has_z:
                result[:, :, iz] = pb_hat
            else:
                result = pb_hat
        
        return result