# post_processing.py
import numpy as np
import matplotlib.pyplot as plt
import os
from .poisson_solver import PoissonSolver
from .tools import get_grids
import imageio

class PostProcessor:
    def __init__(self, diagnostics):
        """
        Initialize the post-processing module.
        
        Parameters:
        -----------
        solution : dict
            Dictionary containing the solution data
            - 't': time array
            - 'moments': dictionary of moment arrays with time as the last dimension
        nx, ny, nz : int
            Number of grid points in each direction
        x, y, z : ndarray
            Coordinate arrays
        """
        self.t = diagnostics.frames['t']
        self.fields = diagnostics.frames['fields']
        self.energy_history = diagnostics.energy_history
        self.enstrophy_history = diagnostics.enstrophy_history

        self.mn2idx = {
            'dens': 0,
            'n': 0,
            'N': 0,
            'upar': 1,
            'zeta': 1,
            'Tpar': 2,
            'Tperp': 3,
            'qpar': 4,
            'qperp': 5,
            'Pparpar': 6,
            'Pperppar': 7,
            'Pperpperp': 8,
            'phi': -1
        }
        grids = get_grids(diagnostics.config.numerical)
        self.x = grids['x']
        self.y = grids['y']
        self.z = grids['z']
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nz = len(self.z)
        
        self.time_indices = np.arange(len(self.t))
        
        # Create output directory
        self.output_dir = 'output/figures'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_moment_data(self, moment_name, time_idx=None):
        if moment_name not in self.mn2idx.keys():
            raise ValueError(f"Unknown moment: {moment_name}")
        
        if time_idx is not None:
            return self.fields[time_idx][self.mn2idx[moment_name]]
        else:
            return self.fields[:,self.mn2idx[moment_name],:,:,:]
    
    def to_real_space(self, data_hat):
        # Check if we have a time dimension
        has_time = len(data_hat.shape) > 3
        
        if has_time:
            nt = data_hat.shape[0]
            data_real = np.zeros((nt, self.nx, self.ny, self.nz))
            
            # Loop over time and z
            for it in range(nt):
                for iz in range(self.nz):
                    data_real[it, :, :, iz] = np.squeeze(np.fft.irfft2(data_hat[it, :, :, iz], s=(self.nx, self.ny)))
        else:
            data_real = np.zeros((self.nx, self.ny, self.nz))
            
            # Loop over z
            for iz in range(self.nz):
                data_real[:, :, iz] = np.fft.irfft2(data_hat[:, :, iz], s=(self.nx, self.ny))
        
        return data_real
    
    def compute_flux_surface_average(self, phi_hat, solver=None):
        """
        Compute the flux surface average of phi.
        
        Parameters:
        -----------
        phi_hat : ndarray
            Electrostatic potential in Fourier space
        solver : PoissonSolver, optional
            Poisson solver with geometry information
            
        Returns:
        --------
        ndarray
            Flux surface averaged phi
        """
        if solver is None:
            # Create a simple geometry class that includes all required attributes
            class SimpleGeometry:
                def __init__(self, kx, ky, z):
                    self.kx = kx
                    self.ky = ky
                    self.z = z
                    self.nkx = len(kx)
                    self.nky = len(ky)
                    self.nz = len(z)
                    self.jacobian = np.ones(self.nz)
                    self.kperp2 = np.ones((self.nkx, self.nky, self.nz))
                    self.l_perp = np.ones((self.nkx, self.nky, self.nz))
                    
                    # Dummy params class to avoid attribute errors
                    class DummyParams:
                        def __init__(self):
                            self.tau = 0.01
                    
                    self.params = DummyParams()
            
            # Create geometry with the required attributes
            geom = SimpleGeometry(
                kx=np.fft.fftfreq(self.nx) if hasattr(self, 'nx') else np.arange(phi_hat.shape[0]),
                ky=np.fft.rfftfreq(self.ny) if hasattr(self, 'ny') else np.arange(phi_hat.shape[1]),
                z=self.z
            )
            solver = PoissonSolver(geom)
        
        # Check if we have a time dimension
        has_time = len(phi_hat.shape) > 3
        
        if has_time:
            nt = phi_hat.shape[-1]
            phi_avg = np.zeros_like(phi_hat)
            
            for it in range(nt):
                phi_avg[..., it] = solver.flux_surf_avg(phi_hat[..., it])
        else:
            phi_avg = solver.flux_surf_avg(phi_hat)
        
        return phi_avg
    

    def create_gif(self, moment_name, time_indices = [], z_idx=0, 
                   clim = [], moviename=[], cbar=False):
        print('Creating GIF movie...')
        # Manage default parameters.
        time_indices = time_indices if time_indices else self.time_indices   
        if clim == 'auto':            
            maxmom = np.max(np.abs(self.get_moment_data(moment_name)))
            clim = [-maxmom, maxmom]        
            
        # Create a directory to store the frames
        os.makedirs(f'tmp_gif_frames', exist_ok=True)
         
        # Generate the frames
        frame_filenames = []
        for time_idx in time_indices:
            frame_filename = f'tmp_gif_frames/gif_frame_{time_idx}.png'
            self.plot_2D_snapshot(moment_name, time_idx, z_idx, cbar=cbar,
                                  clim=clim, filename=frame_filename)
            frame_filenames.append(frame_filename)

        # Create the GIF
        moviename = moviename if moviename else f'{self.output_dir}/{moment_name}_movie.gif'
        print('Compiling ', moviename, '...')
        with imageio.get_writer(moviename, mode='I', duration=0.5) as writer:
            for frame_filename in frame_filenames:
                image = imageio.imread(frame_filename)
                writer.append_data(image)
                
        # Remove the frame files
        for frame_filename in frame_filenames:
            os.remove(frame_filename)
        # Remove the temporary directory
        os.rmdir('tmp_gif_frames')

        print(f'GIF movie created: {moviename}')
        
    def plot_2D_snapshot(self, moment_name, time_idx=0, z_idx=0, 
                         filename='', clim=[], cbar=True):
        """
        Plot a 2D snapshot of a moment at a specific time and z.
        
        Parameters:
        -----------
        moment_name : str
            Name of the moment to plot
        time_idx : int, optional
            Time index to plot
        z_idx : int, optional
            Z index to plot
        """
        # Get the moment data
        data_hat = self.get_moment_data(moment_name, time_idx)
        
        # Transform to real space
        data_real = self.to_real_space(data_hat)
        
        # Select the z plane
        z_plane = data_real[:, :, z_idx]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(z_plane.T, origin='lower', extent=[0, self.x[-1], 0, self.y[-1]], 
                      aspect='auto', cmap='RdBu_r', interpolation='quadric')
        # im = ax.pcolormesh(self.x, self.y, z_plane.T, cmap='RdBu_r')
        if cbar: plt.colorbar(im, ax=ax, label=f'{moment_name}')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{moment_name} at t={self.t[time_idx]:.2f}, z={self.z[z_idx]:.2f}')
        if clim: im.set_clim(clim)
        # Save the figure
        plt.tight_layout()
        filename = filename if filename else f'{self.output_dir}/{moment_name}_t{time_idx}_z{z_idx}.png'
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    
    def plot_time_evolution(self, moment_name, kx_idx=1, ky_idx=1, z_idx=0):
        """
        Plot the time evolution of a specific Fourier mode.
        
        Parameters:
        -----------
        moment_name : str
            Name of the moment to plot
        kx_idx : int, optional
            kx index to plot
        ky_idx : int, optional
            ky index to plot
        z_idx : int, optional
            z index to plot
        """
        # Get the moment data for all times
        data_hat = self.get_moment_data(moment_name)
        
        # Extract the time series for the specified mode
        mode_data = data_hat[:, kx_idx, ky_idx, z_idx]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.t, np.abs(mode_data), 'b-', label='Amplitude')
        ax.plot(self.t, np.real(mode_data), 'r--', label='Real part')
        ax.plot(self.t, np.imag(mode_data), 'g--', label='Imaginary part')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{moment_name}(kx={kx_idx}, ky={ky_idx}, z={z_idx})')
        ax.set_title(f'Time evolution of {moment_name} mode')
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/{moment_name}_mode_kx{kx_idx}_ky{ky_idx}_z{z_idx}.png', dpi=150)
        plt.close(fig)
    
    def plot_energy_evolution(self):
        """
        Plot the time evolution of energy components.
        
        Parameters:
        -----------
        energy_history : list of dict
            List of dictionaries containing energy components at each time
        """
        # Extract time and energy components
        times = self.energy_history['t']
        kinetic = self.energy_history['kinetic']
        thermal = self.energy_history['thermal']
        potential = self.energy_history['potential']
        total = self.energy_history['total']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(times, kinetic, 'r-', label='Kinetic')
        ax.plot(times, thermal, 'g-', label='Thermal')
        ax.plot(times, potential, 'b-', label='Potential')
        ax.plot(times, total, 'k-', linewidth=2, label='Total')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Evolution')
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/energy_evolution.png', dpi=150)
        plt.close(fig)
    
    def plot_enstrophy_evolution(self):
        """
        Plot the time evolution of enstrophy.
        
        Parameters:
        -----------
        enstrophy_history : list of dict
            List of dictionaries containing enstrophy at each time
        """
        
        # Extract time and enstrophy
        times = self.enstrophy_history['t']
        enstrophy = self.enstrophy_history['enstrophy']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(times, enstrophy, 'b-')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Enstrophy')
        ax.set_title('Enstrophy Evolution')
        
        # Save the figure
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/enstrophy_evolution.png', dpi=150)
        plt.close(fig)
    
    def plot_phi_and_avg(self, indices=None):
        """
        Plot phi and its flux surface average for selected time indices.
        
        Parameters:
        -----------
        indices : list of int, optional
            Time indices to plot (default: first, middle, last)
        """
        # Get phi data for all times
        phi_hat = self.get_moment_data('phi')
        
        # Compute flux surface average
        phi_avg_hat = self.compute_flux_surface_average(phi_hat)
        
        # Transform to real space
        phi_real = self.to_real_space(phi_hat)
        phi_avg_real = self.to_real_space(phi_avg_hat)
        
        # Set default indices if not provided
        if indices is None:
            nt = phi_real.shape[0]
            indices = [0, nt//2, -1]
 
        # For each time index
        for idx in indices:
            t_val = self.t[idx]
            
            # Plot for z=0 (mid-plane)
            z_idx = self.nz // 2 if self.nz > 1 else 0
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot phi
            im1 = ax1.imshow(phi_real[idx, :, :, z_idx].T, origin='lower', 
                           extent=[0, self.x[-1], 0, self.y[-1]], cmap='RdBu_r',
                           aspect='auto',interpolation='quadric')
            plt.colorbar(im1, ax=ax1, label='phi')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title(f'phi at t={t_val:.2f}, z={self.z[z_idx]:.2f}')
            
            # Plot phi - <phi>
            phi_diff = phi_real[idx, :, :, z_idx] - phi_avg_real[idx, :, :, z_idx]
            im2 = ax2.imshow(phi_diff.T, origin='lower', 
                           extent=[0, self.x[-1], 0, self.y[-1]], cmap='RdBu_r',
                           aspect='auto',interpolation='quadric')
            plt.colorbar(im2, ax=ax2, label='phi - <phi>')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_title(f'phi - <phi> at t={t_val:.2f}, z={self.z[z_idx]:.2f}')
            
            # Save the figure
            plt.tight_layout()
            fig.savefig(f'{self.output_dir}/phi_and_avg_t{idx}.png', dpi=150)
            plt.close(fig)
            
    def compute_growth_rates(self, moment_name='N', time_window=None, z_idx=0):
        """
        Compute and plot linear growth rates for each (kx, ky) mode.
        
        Parameters:
        -----------
        moment_name : str, optional
            Name of the moment to analyze (default is 'N' for density)
        time_window : tuple, optional
            (start, end) time indices to use for growth rate calculation
            If None, uses the second half of the simulation time
        z_idx : int, optional
            Z index to use for analysis
            
        Returns:
        --------
        ndarray
            2D array of growth rates for each (kx, ky) mode
        """
        # Get the moment data for all times
        data_hat = self.get_moment_data(moment_name)
        
        # Set default time window if not provided (use second half of simulation time)
        if time_window is None:
            nt = data_hat.shape[0]
            time_window = (nt // 2, nt - 1)
        
        # Extract time range
        t_start, t_end = time_window
        times = self.t[t_start:t_end+1]
        
        # Initialize growth rate array
        growth_rates = np.zeros((self.nx, self.ny // 2 + 1))
        max_amplitude = np.zeros((self.nx, self.ny // 2 + 1))
        
        # Loop over all modes
        for ikx in range(self.nx):
            for iky in range(self.ny // 2 + 1):
                # Extract mode amplitude over time
                mode_data = data_hat[t_start:t_end+1, ikx, iky, z_idx]
                amplitude = np.abs(mode_data)
                
                # Store maximum amplitude
                max_amplitude[ikx, iky] = np.max(amplitude)
                
                # Skip modes with very small amplitude to avoid numerical issues
                if max_amplitude[ikx, iky] < 1e-10:
                    growth_rates[ikx, iky] = 0.0
                    continue
                
                # Compute growth rate using linear fit in log space
                if np.all(amplitude > 0):  # Ensure all amplitudes are positive
                    try:
                        # Use only the growing part for the fit
                        # Take log of amplitude for linear fit
                        log_amplitude = np.log(amplitude)
                        
                        # Linear fit: log(A) = γt + c
                        # Fit a line to the log of the amplitude
                        polyfit = np.polyfit(times, log_amplitude, 1)
                        growth_rates[ikx, iky] = polyfit[0]  # Slope = growth rate
                    except:
                        growth_rates[ikx, iky] = 0.0
                else:
                    growth_rates[ikx, iky] = 0.0
        
        # Plot the growth rate spectrum
        self._plot_growth_rate_spectrum(growth_rates, max_amplitude, moment_name)
        
        # Plot a few individual modes to verify the growth rate calculation
        self._plot_mode_growth_examples(data_hat, growth_rates, moment_name, time_window, z_idx)
        
        return growth_rates

    def _plot_growth_rate_spectrum(self, growth_rates, max_amplitude, moment_name):
        """
        Plot the 2D spectrum of growth rates.
        
        Parameters:
        -----------
        growth_rates : ndarray
            2D array of growth rates
        max_amplitude : ndarray
            2D array of maximum amplitudes
        moment_name : str
            Name of the moment being analyzed
        """
        # Create the growth rate plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get the actual wavenumbers for the axes
        kx_values = np.fft.fftfreq(self.nx, d=self.x[1]-self.x[0]) * 2 * np.pi
        ky_values = np.fft.rfftfreq(self.ny, d=self.y[1]-self.y[0]) * 2 * np.pi
        
        # Create extent for imshow based on actual wavenumbers
        extent = [ky_values[0], ky_values[-1], kx_values[0], kx_values[-1]]
        
        # Only show positive growth rates for clarity
        growth_rates_plot = growth_rates.copy()
        growth_rates_plot[growth_rates_plot < 0] = 0
        
        # Plot the growth rate spectrum
        # im = ax.imshow(growth_rates_plot, origin='lower', extent=extent, 
        #             aspect='auto', cmap='plasma')
        KX, KY = np.meshgrid(np.sort(kx_values), ky_values, indexing='ij')
        growth_rates_plot = np.fft.fftshift(growth_rates_plot, axes=0)
        im = ax.pcolormesh(KX, KY, growth_rates_plot, cmap='plasma')
        plt.colorbar(im, ax=ax, label='Growth Rate (γ)')
        
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title(f'Linear Growth Rate Spectrum for {moment_name}')
        
        # Save the figure
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/{moment_name}_growth_rate_spectrum.png', dpi=150)
        plt.close(fig)
        
        # Also plot growth rate with amplitude as marker size
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a meshgrid for scatter plot
        kx_grid, ky_grid = np.meshgrid(kx_values, ky_values, indexing='ij')
        
        # Only include modes with positive growth rates and significant amplitude
        mask = (growth_rates > 0) & (max_amplitude > 1e-6)
        
        # Normalize marker sizes based on amplitude
        if np.any(mask):
            size_scale = max_amplitude[mask] / np.max(max_amplitude[mask]) * 100
            scatter = ax.scatter(ky_grid[mask], kx_grid[mask], 
                            c=growth_rates[mask], cmap='plasma', 
                            s=size_scale, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Growth Rate (γ)')
        else:
            ax.text(0.5, 0.5, 'No growing modes found', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        
        ax.set_xlabel('ky')
        ax.set_ylabel('kx')
        ax.set_title(f'Growth Rates for {moment_name} (marker size = amplitude)')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Save the figure
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/{moment_name}_growth_rate_amplitude.png', dpi=150)
        plt.close(fig)

    def _plot_mode_growth_examples(self, data_hat, growth_rates, moment_name, time_window, z_idx):
        """
        Plot examples of mode growth for verification.
        
        Parameters:
        -----------
        data_hat : ndarray
            Full spectral data array
        growth_rates : ndarray
            Computed growth rates
        moment_name : str
            Name of the moment being analyzed
        time_window : tuple
            (start, end) time indices
        z_idx : int
            Z index used for analysis
        """
        # Find a few interesting modes to plot
        # 1. Fastest growing mode
        max_idx = np.unravel_index(np.argmax(growth_rates), growth_rates.shape)
        
        # 2. Another growing mode (if exists)
        other_growing = None
        sorted_indices = np.argsort(growth_rates.flatten())[::-1]
        if len(sorted_indices) > 1:
            second_idx = sorted_indices[1]
            other_growing = np.unravel_index(second_idx, growth_rates.shape)
        
        # 3. A stable/damped mode (negative growth rate)
        min_idx = None
        if np.any(growth_rates < 0):
            min_idx = np.unravel_index(np.argmin(growth_rates), growth_rates.shape)
        
        # Extract the time range
        t_start, t_end = time_window
        times = self.t[t_start:t_end+1]
        
        # Create a figure for mode examples
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        
        # Plot the fastest growing mode
        ikx, iky = max_idx
        mode_data = data_hat[t_start:t_end+1, ikx, iky, z_idx]
        amplitude = np.abs(mode_data)
        
        # Plot amplitude in log scale
        axs[0].semilogy(times, amplitude, 'b-', label='Amplitude')
        
        # Overlay the fitted exponential growth
        growth_rate = growth_rates[ikx, iky]
        A0 = amplitude[0]
        fitted_amplitude = A0 * np.exp(growth_rate * (times - times[0]))
        axs[0].semilogy(times, fitted_amplitude, 'r--', 
                    label=f'Fit: γ = {growth_rate:.4f}')
        
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Amplitude (log scale)')
        axs[0].set_title(f'Fastest Growing Mode: (kx={ikx}, ky={iky})')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot another growing mode if available
        if other_growing:
            ikx, iky = other_growing
            mode_data = data_hat[t_start:t_end+1, ikx, iky, z_idx]
            amplitude = np.abs(mode_data)
            
            axs[1].semilogy(times, amplitude, 'b-', label='Amplitude')
            
            growth_rate = growth_rates[ikx, iky]
            A0 = amplitude[0]
            fitted_amplitude = A0 * np.exp(growth_rate * (times - times[0]))
            axs[1].semilogy(times, fitted_amplitude, 'r--', 
                        label=f'Fit: γ = {growth_rate:.4f}')
            
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Amplitude (log scale)')
            axs[1].set_title(f'Another Growing Mode: (kx={ikx}, ky={iky})')
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].text(0.5, 0.5, 'No other growing mode found', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axs[1].transAxes, fontsize=12)
        
        # Plot damped mode if available
        if min_idx:
            ikx, iky = min_idx
            mode_data = data_hat[t_start:t_end+1, ikx, iky, z_idx]
            amplitude = np.abs(mode_data)
            
            axs[2].semilogy(times, amplitude, 'b-', label='Amplitude')
            
            growth_rate = growth_rates[ikx, iky]
            A0 = amplitude[0]
            fitted_amplitude = A0 * np.exp(growth_rate * (times - times[0]))
            axs[2].semilogy(times, fitted_amplitude, 'r--', 
                        label=f'Fit: γ = {growth_rate:.4f}')
            
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Amplitude (log scale)')
            axs[2].set_title(f'Damped Mode: (kx={ikx}, ky={iky})')
            axs[2].legend()
            axs[2].grid(True)
        else:
            axs[2].text(0.5, 0.5, 'No damped mode found', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axs[2].transAxes, fontsize=12)
        
        # Plot the zonal mode (kx=any, ky=0) which is often important in plasma turbulence
        ky0_idx = 0
        kx_middle = self.nx // 4  # Use a mid-range kx
        mode_data = data_hat[t_start:t_end+1, kx_middle, ky0_idx, z_idx]
        amplitude = np.abs(mode_data)
        
        axs[3].semilogy(times, amplitude, 'b-', label='Amplitude')
        
        growth_rate = growth_rates[kx_middle, ky0_idx]
        A0 = amplitude[0]
        fitted_amplitude = A0 * np.exp(growth_rate * (times - times[0]))
        axs[3].semilogy(times, fitted_amplitude, 'r--', 
                    label=f'Fit: γ = {growth_rate:.4f}')
        
        axs[3].set_xlabel('Time')
        axs[3].set_ylabel('Amplitude (log scale)')
        axs[3].set_title(f'Zonal Mode: (kx={kx_middle}, ky=0)')
        axs[3].legend()
        axs[3].grid(True)
        
        # Save the figure
        plt.tight_layout()
        fig.savefig(f'{self.output_dir}/{moment_name}_mode_growth_examples.png', dpi=150)
        plt.close(fig)