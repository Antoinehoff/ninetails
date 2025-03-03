# diagnostics.py
import numpy as np

class Diagnostics:
    def __init__(self, grid, params):
        self.grid = grid
        self.params = params
        self.energy_history = []
        self.enstrophy_history = []
        
    def compute_energy(self, moments, phi):
        """Compute total energy from moments and potential"""
        N, u_par, T_par, T_perp = moments[:4]
        
        # Kinetic energy
        E_kin = 0.5 * np.mean(np.abs(u_par)**2)
        
        # Thermal energy
        E_therm = 0.5 * np.mean(np.abs(T_par)**2 + np.abs(T_perp)**2)
        
        # Potential energy
        E_pot = 0.5 * np.mean(np.abs(phi)**2)
        
        return E_kin, E_therm, E_pot
        
    def compute_enstrophy(self, phi, grid):
        """Compute enstrophy from potential"""
        # Compute vorticity in Fourier space
        # Access the kx and ky from the dictionary correctly
        kx = grid['kx']
        ky = grid['ky']
        
        # Create 2D meshgrid for kx and ky if they are 1D arrays
        if kx.ndim == 1 and ky.ndim == 1:
            kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
        else:
            kx_2d, ky_2d = kx, ky
            
        # Reshape for broadcasting correctly with phi dimensions
        kx_shape = [1] * phi.ndim
        ky_shape = [1] * phi.ndim
        kx_shape[0] = kx_2d.shape[0]
        kx_shape[1] = kx_2d.shape[1]
        ky_shape[0] = ky_2d.shape[0]
        ky_shape[1] = ky_2d.shape[1]
        
        kx_reshaped = kx_2d.reshape(kx_shape)
        ky_reshaped = ky_2d.reshape(ky_shape)
        
        # Compute vorticity (Laplacian of phi in Fourier space)
        vort_hat = -(kx_reshaped**2 + ky_reshaped**2) * phi
        
        # Compute enstrophy
        enstrophy = 0.5 * np.mean(np.abs(vort_hat)**2)
        
        return enstrophy
        
    def update(self, t, moments, phi):
        """Update diagnostics at time t"""
        E_kin, E_therm, E_pot = self.compute_energy(moments, phi)
        enstrophy = self.compute_enstrophy(phi, self.grid)
        
        self.energy_history.append({
            't': t,
            'kinetic': E_kin,
            'thermal': E_therm,
            'potential': E_pot,
            'total': E_kin + E_therm + E_pot
        })
        
        self.enstrophy_history.append({
            't': t,
            'enstrophy': enstrophy
        })