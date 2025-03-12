# post_processing.py
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from .postprocessor import PostProcessor

class Plotter:

    def __init__(self, simulation, output_dir=None, figsize=(6, 4)):
        self.simulation = simulation
        self.figsize = figsize
        # Create output directory
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

    def create_gif(self, moment_name, time_indices = [], z_idx=0, 
                   clim = [], moviename=[], cbar=False):
        print('Creating GIF movie...')
        # Manage default parameters.
        time_indices = time_indices if time_indices else self.simulation.diagnostics.nframes   
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
        moviename = moviename if moviename else f'{self.simulation.config.output_dir}/{moment_name}_movie.gif'
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
        
    def energy_evolution(self, filename=''):
        """
        Plot the time evolution of energy components.
        
        Parameters:
        -----------
        energy_history : list of dict
            List of dictionaries containing energy components at each time
        """
        # Extract time and energy components
        times = self.simulation.diagnostics.integrated['t']
        kinetic = self.simulation.diagnostics.integrated['Ekin']
        thermal = self.simulation.diagnostics.integrated['Eth']
        potential = self.simulation.diagnostics.integrated['Epot']
        total = self.simulation.diagnostics.integrated['Etot']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
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
        if self.simulation.config.output_dir:
            fig.savefig(f'{self.simulation.config.output_dir}/energy_evolution.png', dpi=150)
        if filename:
            fig.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()
        
    def growth_rates(self, moment_name='N', time_window=None, z_idx=0, filename=''):
        
        time, field = self.simulation.diagnostics.get_moment_data(moment_name)

        
        # Only show positive growth rates for clarity
        growth_rates, max_amplitude = PostProcessor.compute_growth_rates(self.simulation, time, field, time_window, z_idx)
        growth_rates[growth_rates < 0] = 0
        

        # Create the growth rate plot
        fig, ax = plt.subplots(figsize=(10, 8))

        kx = self.simulation.kxgrids[0]
        ky = self.simulation.kygrids[1]
        # Plot the growth rate spectrum
        kx_grid, ky_grid = np.meshgrid(np.sort(kx), ky, indexing='ij')
        growth_rates_plot = np.fft.fftshift(growth_rates_plot, axes=0)
        im = ax.pcolormesh(kx_grid, ky_grid, growth_rates_plot, cmap='plasma')
        plt.colorbar(im, ax=ax, label='Growth Rate (γ)')
        
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title(f'Linear Growth Rate Spectrum for {moment_name}')
        
        # Save the figure
        plt.tight_layout()
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/{moment_name}_growth_rate_spectrum.png', dpi=150)
        if filename:
            fig.savefig(filename, dpi=150)
        plt.close(fig)
        
        # Also plot growth rate with amplitude as marker size
        fig, ax = plt.subplots(figsize=(10, 8))
        
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
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/{moment_name}_growth_rate_amplitude.png', dpi=150)
        if filename:
            fig.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def _plot_mode_growth_examples(self, data_hat, growth_rates, moment_name, time_window, z_idx, filename=''):
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
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/{moment_name}_mode_growth_examples.png', dpi=150)
        if filename:
            fig.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()