# diagnostics.py
import numpy as np
from .postprocessor import PostProcessor
from .tools import get_grids

class Diagnostics:
    def __init__(self, config = None):
        
        self.config = config
        
        self.grid = get_grids(config.numerical)
        
        self.nframes = config.nframes
        self.frames = {}
        self.frames['t'] = []
        self.frames['fields'] = []
        
        self.energy_history = {}
        self.energy_history['t'] = []
        self.energy_history['kinetic'] = []
        self.energy_history['thermal'] = []
        self.energy_history['potential'] = []
        self.energy_history['total'] = []
        
        self.enstrophy_history = {}
        self.enstrophy_history['t'] = []
        self.enstrophy_history['enstrophy'] = []
        
        # compute the time where to save the data
        self.t_frame_diag = np.linspace(0, config.numerical.max_time, config.nframes)
        self.dt_frame_diag = self.t_frame_diag[1] - self.t_frame_diag[0]
        self.t_int_diag = np.linspace(0, config.numerical.max_time, config.nframes*10)
        self.dt_int_diag = self.t_int_diag[1] - self.t_int_diag[0]

    def update(self, t, y):
        """Update diagnostics at time t"""
        
        if t >= self.t_frame_diag[0]:
            self.save_frame(t, y)
            self.t_frame_diag = self.t_frame_diag[1:]
        
        if t >= self.t_int_diag[0]:
            E_kin, E_therm, E_pot = self.compute_energy(y)
            enstrophy = self.compute_enstrophy(y[-1], self.grid)

            self.energy_history['t'].append(t)
            self.energy_history['kinetic'].append(E_kin)
            self.energy_history['thermal'].append(E_therm)
            self.energy_history['potential'].append(E_pot)
            self.energy_history['total'].append(E_kin + E_therm + E_pot)

            self.enstrophy_history['t'].append(t)
            self.enstrophy_history['enstrophy'].append(enstrophy)
            
            print(f"t = {t:.2e}, E_tot = {E_kin + E_therm + E_pot:.2e}")
            
            self.t_int_diag = self.t_int_diag[1:]
            
    def finalize(self):
        # convert lists to numpy arrays
        for key in self.frames.keys():
            self.frames[key] = np.array(self.frames[key])
        for key in self.energy_history.keys():
            self.energy_history[key] = np.array(self.energy_history[key])
        for key in self.enstrophy_history.keys():
            self.enstrophy_history[key] = np.array(self.enstrophy_history[key])
        
    def save_frame(self, t, y):
        """Save a frame of the moments and potential"""
        self.frames['t'].append(t)
        self.frames['fields'].append(y)
        self.nframes += 1
        
    def compute_energy(self, y):
        """Compute total energy from moments and potential"""
        N, u_par, T_par, T_perp = y[:4]
        phi = y[-1]
        
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
