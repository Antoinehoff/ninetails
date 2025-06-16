# post_processing.py
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from .postprocessor import PostProcessor

class Plotter:
    '''
    Class to create plots and movies from simulation data.
    
    Parameters:
    -----------
    simulation : Simulation
        Simulation object
    output_dir : str [optional]
        Directory to save the plots and movies
    figsize : tuple
        Size of the figure
        '''
    def __init__(self, simulation, output_dir=None, figsize=(6, 4)):
        self.simulation = simulation
        self.figsize = figsize
        # Create output directory
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = None

    def create_gif(self, moment_name, time_indices = [], z_idx=0, 
                   clim = [], moviename=[], cbar=False):
        print('Creating GIF movie...')
        # Manage default parameters.
        time_indices = time_indices if time_indices \
            else range(self.simulation.diagnostics.nframes)   
        if clim == 'auto':            
            maxmom = np.max(np.abs(self.get_moment_data(moment_name)))
            clim = [-maxmom, maxmom]        
            
        # Create a directory to store the frames
        os.makedirs(f'tmp_gif_frames', exist_ok=True)
        
        # Generate the frames
        frame_filenames = []
        for time_idx in time_indices:
            frame_filename = f'tmp_gif_frames/gif_frame_{time_idx}.png'
            self.snapshot(moment_name, time_idx, z_idx, cbar=cbar,
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
        
        plt.tight_layout()
        plt.show()
        # Save the figure
        if self.simulation.config.output_dir:
            fig.savefig(f'{self.simulation.config.output_dir}/energy_evolution.png', dpi=150)
        if filename:
            fig.savefig(filename, dpi=150)
            plt.close(fig)
        
    def growth_rates(self, moment_name='N', time_window=None, z_idx=0, cut_direction=None, slice_idx=None, show_error=False, filename=''):
        time, field = self.simulation.diagnostics.get_moment_data(moment_name)
        
        # Compute growth rates and errors
        growth_rates, errors = PostProcessor.compute_growth_rates(
            self.simulation, time, field, time_window, z_idx )
        growth_rates[growth_rates < 0] = 0
        
        # Get the wavenumber grids
        kx = self.simulation.kgrids[0]
        ky = self.simulation.kgrids[1]

        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)

        if cut_direction is None or cut_direction == '':
            # Plot 2D growth rate spectrum
            kx = np.sort(kx)
            kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
            growth_rates = np.fft.fftshift(growth_rates, axes=0)
            im = ax.pcolormesh(kx_grid, ky_grid, growth_rates, cmap='bwr')
            plt.colorbar(im, ax=ax, label='Growth Rate (γ)')
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_title(f'Linear Growth Rate Spectrum for {moment_name}')
        else:
            # Plot 1D cut
            if cut_direction.lower() == 'kx':
                if slice_idx is None:
                    slice_idx = 0
                rates = growth_rates[slice_idx]
                err = errors[:, slice_idx] if show_error else None
                x_values = kx
                xlabel = 'kx'
                title_extra = f' at ky={ky[slice_idx]:.2f}'
            elif cut_direction.lower() == 'ky':
                if slice_idx is None:
                    slice_idx = 0
                rates = growth_rates[slice_idx, :]
                err = errors[slice_idx, :] if show_error else None
                x_values = ky
                xlabel = 'ky'
                title_extra = f' at kx={kx[slice_idx]:.2f}'
            else:
                raise ValueError("cut_direction must be either 'kx' or 'ky'")

            if show_error:
                ax.errorbar(x_values, rates, yerr=err, fmt='o-', capsize=3)
            else:
                ax.plot(x_values, rates, 'o-')
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Growth Rate (γ)')
            ax.set_title(f'Linear Growth Rate for {moment_name}{title_extra}')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        # Save the figure
        if self.output_dir:
            base_name = f'{moment_name}_growth_rate'
            if cut_direction:
                base_name += f'_{cut_direction.lower()}_cut'
            fig.savefig(f'{self.output_dir}/{base_name}.png', dpi=150)
            plt.close(fig)
        if filename:
            fig.savefig(filename, dpi=150)
            plt.close(fig)
 
    def snapshot(self, moment_name='N', time_idx=-1, z_idx=0, plane='xy', cbar=False, clim=[], filename=''):
        # Extract the data
        time, field = self.simulation.diagnostics.get_moment_data(moment_name, time_idx)
        
        if plane == 'xy':
            field = PostProcessor.to_real_space(self.simulation,field)

        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize if hasattr(self, 'figsize') else (6, 4))
        
        # Plot the data
        im = ax.imshow(field[:, :, z_idx].T, origin='lower', cmap='RdBu')
        if cbar: plt.colorbar(im, ax=ax, label=moment_name)
        if clim: im.set_clim(clim)
        
        ax.set_title(f'{moment_name} at t = {time:.2f}, amp: {np.max(np.abs(field[:, :, z_idx])):.2e}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        plt.tight_layout()
        plt.show()
        # Save the figure
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/{moment_name}_t{time_idx}.png', dpi=150)
            plt.close(fig)
        if filename:
            fig.savefig(filename, dpi=150)
            plt.close(fig)
            
    def mode_amplitude_evolution(self, moment_name='N', z_idx=0, filename=''):
        time, field = self.simulation.diagnostics.get_moment_data(moment_name)
        mode_amplitudes_kx, mode_amplitudes_ky = PostProcessor.compute_mode_amplitude_evolution(
            self.simulation, time, field, z_idx
        )

        fig, axs = plt.subplots(2, 1, figsize=self.figsize if hasattr(self, 'figsize') else (6, 8))

        # Use colormap for coloring the curves
        cmap_kx = plt.cm.viridis(np.linspace(0, 1, mode_amplitudes_kx.shape[1]))
        cmap_ky = plt.cm.viridis(np.linspace(0, 1, mode_amplitudes_ky.shape[1]))

        # Plot mode amplitudes for all kx with ky=0
        for kx_idx in range(mode_amplitudes_kx.shape[1]):
            axs[0].plot(time, mode_amplitudes_kx[:, kx_idx], color=cmap_kx[kx_idx])
        axs[0].set_title(f'Mode Amplitude Evolution (ky=0) for {moment_name}')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Amplitude')

        # Add colorbar for kx modes
        sm_kx = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=mode_amplitudes_kx.shape[1] - 1))
        sm_kx.set_array([])
        cbar_kx = fig.colorbar(sm_kx, ax=axs[0], orientation='vertical')
        cbar_kx.set_label('kx Mode Number')

        # Plot mode amplitudes for all ky with kx=0
        for ky_idx in range(mode_amplitudes_ky.shape[1]):
            axs[1].plot(time, mode_amplitudes_ky[:, ky_idx], color=cmap_ky[ky_idx])
        axs[1].set_title(f'Mode Amplitude Evolution (kx=0) for {moment_name}')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Amplitude')

        # Add colorbar for ky modes
        sm_ky = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=mode_amplitudes_ky.shape[1] - 1))
        sm_ky.set_array([])
        cbar_ky = fig.colorbar(sm_ky, ax=axs[1], orientation='vertical')
        cbar_ky.set_label('ky Mode Number')

        plt.tight_layout()
        plt.show()

        # Save the figure
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/{moment_name}_mode_amplitude_evolution.png', dpi=150)
        if filename:
            fig.savefig(filename, dpi=150)
