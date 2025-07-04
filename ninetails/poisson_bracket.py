# poisson_bracket.py
import numpy as np
from .fastfouriertransform import FastFourierTransform

class PoissonBracket:
    def __init__(self, kx, ky, AA_method='filtering'):
        """
        Initialize the Poisson bracket calculator.
        
        Parameters:
        -----------
        kx : ndarray
            Radial wavenumbers (2D array)
        ky : ndarray
            Binormal wavenumbers (2D array)
        method : str
            Method to use for anti-aliasing. Options are 'filtering' and 'padding'
        """
        self.AA_method = AA_method
        self.kx = kx
        self.ky = ky
        self.nkx, self.nky = kx.shape
        self.nx = self.nkx
        self.ny = 2 * self.nky - 1
        self.mult_factor = 1#2*np.pi/kx[1,0] * 2*np.pi/ky[0,1]
        
        # Calculate dealiasing pad sizes (using 3/2 rule)
        self.nx_pad = int(self.nx * 3 / 2)
        self.ny_pad = int(self.ny * 3 / 2)
                        
        # Select the anti-aliasing method
        if self.AA_method == 'filtering':
            # create the filtering array
            kx_cut = 2./3. * np.max(np.abs(kx))
            ky_cut = 2./3. * np.max(np.abs(ky))
            self.filter = np.ones((self.nkx, self.nky))
            self.filter[np.abs(kx) >= kx_cut] = 0.0
            self.filter[np.abs(ky) >= ky_cut] = 0.0
            self.compute = self.compute_filtering
        elif self.AA_method == 'padding':
            self.compute = self.compute_padding
        else:
            raise ValueError("Invalid anti-aliasing method")

        self.norm='ortho'
        self.axes=(-2, -1)
        self.size =(self.nx, self.ny)
        self.FFT = FastFourierTransform(norm=self.norm, axes=self.axes, size=self.size)
        
        # Pre-allocate work arrays to avoid allocations in compute
        self._work_a_filtered = np.zeros((self.nkx, self.nky), dtype=np.complex128)
        self._work_b_filtered = np.zeros((self.nkx, self.nky), dtype=np.complex128)
        self._work_da_dx = np.zeros((self.nkx, self.nky), dtype=np.complex128)
        self._work_da_dy = np.zeros((self.nkx, self.nky), dtype=np.complex128)
        self._work_db_dx = np.zeros((self.nkx, self.nky), dtype=np.complex128)
        self._work_db_dy = np.zeros((self.nkx, self.nky), dtype=np.complex128)
        self._work_conv1 = np.zeros((self.nkx, self.nky), dtype=np.complex128)
        self._work_conv2 = np.zeros((self.nkx, self.nky), dtype=np.complex128)
        self._work_result = np.zeros((self.nkx, self.nky), dtype=np.complex128)
        
    
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
    
    def convolution_theorem(self, f, g, result=None):                            
        """
        Compute convolution using FFT. If result is provided, store output there.
        """
        conv = self.FFT.rfft2(self.FFT.irfft2(f) * self.FFT.irfft2(g))
        if result is not None:
            result[:] = conv
            return result
        else:
            return conv
        
    def compute_filtering(self, a_hat, b_hat):
        
        if a_hat.shape != b_hat.shape:
            raise ValueError("Input arrays must have the same shape")
        
        # Determine if we have a z dimension
        has_z = len(a_hat.shape) > 2
        z_dim = a_hat.shape[2] if has_z else 1
        
        # Initialize the result array
        result_shape = a_hat.shape
        if has_z:
            result = np.zeros(result_shape, dtype=np.complex128)
        else:
            result = self._work_result
        
        ky_pb_2d = self.kx
        kx_pb_2d = self.ky
        
        # Loop over z dimension if present
        for iz in range(z_dim):
            # Extract 2D slices if we have a z dimension
            a_hat_2d = a_hat[:, :, iz] if has_z else a_hat
            b_hat_2d = b_hat[:, :, iz] if has_z else b_hat

            # Apply the 2/3 rule filter (use pre-allocated arrays)
            self._work_a_filtered[:] = self.filter * a_hat_2d
            self._work_b_filtered[:] = self.filter * b_hat_2d
            
            # Compute derivatives in Fourier space (use pre-allocated arrays)
            self._work_da_dx[:] = 1j * kx_pb_2d * self._work_a_filtered
            self._work_da_dy[:] = 1j * ky_pb_2d * self._work_a_filtered
            self._work_db_dx[:] = 1j * kx_pb_2d * self._work_b_filtered
            self._work_db_dy[:] = 1j * ky_pb_2d * self._work_b_filtered
            
            # Compute Poisson bracket in Fourier space using convolution theorem
            self.convolution_theorem(self._work_da_dx, self._work_db_dy, self._work_conv1)
            self.convolution_theorem(self._work_da_dy, self._work_db_dx, self._work_conv2)
            
            self._work_result[:] = self._work_conv1 - self._work_conv2
            
            # Filter the result
            self._work_result[:] = self.filter * self._work_result
            
            # Store the result
            if has_z:
                result[:, :, iz] = self._work_result
            else:
                result = self._work_result
        
        return result
    
       
    def compute_padding(self, a_hat, b_hat):
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
        kx_pb_2d = np.zeros((self.nx_pad, self.ny_pad))
        ky_pb_2d = np.zeros((self.nx_pad, self.ny_pad))
        
        # Fill padded wavenumbers
        pad_x = (self.nx_pad - self.nx) // 2
        pad_y = (self.ny_pad - self.ny) // 2
        kx_pb_2d[pad_x:pad_x+self.nx, pad_y:pad_y+self.ny] = self.kx
        ky_pb_2d[pad_x:pad_x+self.nx, pad_y:pad_y+self.ny] = self.ky
        
        # Loop over z dimension if present
        for iz in range(z_dim):
            # Extract 2D slices if we have a z dimension
            a_hat_2d = a_hat[:, :, iz] if has_z else a_hat
            b_hat_2d = b_hat[:, :, iz] if has_z else b_hat
            
            # Pad the arrays for anti-aliasing
            a_hat_pb = self.pad(a_hat_2d)
            b_hat_pb = self.pad(b_hat_2d)
            
            # Compute derivatives in Fourier space
            da_dx_pb = 1j * kx_pb_2d * a_hat_pb
            da_dy_pb = 1j * ky_pb_2d * a_hat_pb
            db_dx_pb = 1j * kx_pb_2d * b_hat_pb
            db_dy_pb = 1j * ky_pb_2d * b_hat_pb
            
            # Compute Poisson bracket in Fourier space using convolution theorem
            pb_hat_pb = self.convolution_theorem(da_dx_pb, db_dy_pb) \
                       -self.convolution_theorem(da_dy_pb, db_dx_pb)
            
            # Unpad the result
            pb_hat = self.unpad(pb_hat_pb)
            
            # Store the result
            if has_z:
                result[:, :, iz] = pb_hat
            else:
                result = pb_hat
        
        return result