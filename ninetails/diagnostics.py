# diagnostics.py
import numpy as np
import matplotlib.pyplot as plt

from .postprocessor import PostProcessor
from .tools import get_grids

class Diagnostics:
    def __init__(self, config = None, show_frame=False):
        
        self.config = config
        
        self.grid = get_grids(config.numerical)
        
        self.nframes = config.nframes
        self.frames = {}
        self.frames['t'] = []
        self.frames['fields'] = []
        self.show_frame = show_frame

        self.integrated = {}
        self.integrated['t'] = []
        self.integrated['Ekin'] = []
        self.integrated['Eth'] = []
        self.integrated['Epot'] = []
        self.integrated['Etot'] = []
        self.integrated['enstrophy'] = []
        
        # compute the time where to save the data
        self.t_frame_diag = np.linspace(0, config.numerical.max_time, config.nframes)
        self.dt_frame_diag = self.t_frame_diag[1] - self.t_frame_diag[0]
        self.t_int_diag = np.linspace(0, config.numerical.max_time, config.nframes*10)
        self.dt_int_diag = self.t_int_diag[1] - self.t_int_diag[0]

        self.mn2idx = {
            'dens': 0,
            'N00' : 0,
            'n': 0,
            'N': 0,
            'upar': 1,
            'N10' : 1,
            'zeta': 1,
            'Tpar': 2,
            'N20' : 2,
            'Tperp': 3,
            'N01' : 3,
            'qpar': 4,
            'N30' : 4,
            'qperp': 5,
            'N11' : 5,
            'Pparpar': 6,
            'N40' : 6,
            'Pperppar': 7,
            'N21' : 7,
            'Pperpperp': 8,
            'N02' : 8,
            'phi': -1
        }

        self.uidx = 0

    def update(self, t, y):
        """Update diagnostics at time t"""
        if self.uidx == 0: self.init()
        self.uidx += 1

        if t >= self.t_frame_diag[0]:
            self.save_frame(t, y)
            self.t_frame_diag = self.t_frame_diag[1:]
            self.show_last_frame(t,y)
        
        if t >= self.t_int_diag[0]:
            E_kin, E_therm, E_pot, enstrophy = self.compute_integrated(y, self.grid)

            self.integrated['t'].append(t)
            self.integrated['Ekin'].append(E_kin)
            self.integrated['Eth'].append(E_therm)
            self.integrated['Epot'].append(E_pot)
            self.integrated['Etot'].append(E_kin + E_therm + E_pot)
            self.integrated['enstrophy'].append(enstrophy)
                        
            # print terminal output into a file
            with open('std.out', 'a') as f:
                f.write(f"t = {t:.2e}, E_tot = {E_kin + E_therm + E_pot:.2e}\n")

            self.t_int_diag = self.t_int_diag[1:]
            
    def init(self):
        if self.config.follow_frame:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111)
            plt.ion()
            plt.show()

    def finalize(self):
        # convert lists to numpy arrays
        for key in self.frames.keys():
            self.frames[key] = np.array(self.frames[key])
        for key in self.integrated.keys():
            self.integrated[key] = np.array(self.integrated[key])
        self.nframes = len(self.frames['t'])
        self.nintegrated = len(self.integrated['t'])


    def save_frame(self, t, y):
        """Save a frame of the moments and potential"""
        self.frames['t'].append(t)
        self.frames['fields'].append(y)
        self.nframes += 1
        if self.config.follow_frame:
            self.show_last_frame(t, y)

    def show_last_frame(self, t, y):
        """Show the last frame"""
        if self.show_frame:
            #self.ax.clear()
            self.ax.plot(self.grid['x'], y[0,0,0,0], label='N')
            self.ax.plot(self.grid['x'], y[1,0,0,0], label='u')
            self.ax.plot(self.grid['x'], y[2,0,0,0], label='T')    
            self.ax.plot(self.grid['x'], y[3,0,0,0], label='phi')
            self.ax.legend()
            self.ax.set_title(f"t = {t:.2e}")
            plt.pause(0.1)

    def compute_integrated(self, y, grid):
        """Compute total energy from moments and potential"""
        N, u_par, T_par, T_perp = y[:4]
        phi = y[-1]
        
        # Kinetic energy
        E_kin = 0.5 * np.mean(np.abs(u_par)**2)
        # Thermal energy
        E_therm = 0.5 * np.mean(np.abs(T_par)**2 + np.abs(T_perp)**2)
        # Potential energy
        E_pot = 0.5 * np.mean(np.abs(phi)**2)
        
        # Compute enstrophy
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
        
        return E_kin, E_therm, E_pot, enstrophy

    def get_moment_data(self, moment_name, time_idx=None):
        if moment_name not in self.mn2idx.keys():
            raise ValueError(f"Unknown moment: {moment_name}")
        
        if time_idx is not None:
            t = self.frames['t'][time_idx]
            return t, self.frames['fields'][time_idx][self.mn2idx[moment_name],:,:,:]
        else:
            t = self.frames['t']
            return t, self.frames['fields'][:,self.mn2idx[moment_name],:,:,:]
        
    def get_history_data(self, data_name):
        if data_name == 'enstrophy':
            return self.enstrophy_history[data_name], self.enstrophy_history['t']
        elif data_name in ['kinetic', 'thermal', 'potential', 'total']:
            return self.integrated[data_name], self.integrated['t']
        else:
            raise ValueError(f"Unknown data: {data_name}, available data are: 'enstrophy', 'kinetic', 'thermal', 'potential', 'total'")