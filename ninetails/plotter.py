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

        kx = self.simulation.kgrids[0]
        ky = self.simulation.kgrids[1]
        # Plot the growth rate spectrum
        kx_grid, ky_grid = np.meshgrid(np.sort(kx), ky, indexing='ij')
        growth_rates = np.fft.fftshift(growth_rates, axes=0)
        im = ax.pcolormesh(kx_grid, ky_grid, growth_rates, cmap='plasma')
        plt.colorbar(im, ax=ax, label='Growth Rate (γ)')
        
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title(f'Linear Growth Rate Spectrum for {moment_name}')
        
        # Save the figure
        plt.tight_layout()
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/{moment_name}_growth_rate_spectrum.png', dpi=150)
            plt.close(fig)
        if filename:
            fig.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()
 
    def snapshot(self, moment_name='N', time_idx=0, z_idx=0, plane='xy', cbar=False, clim=[], filename=''):
        # Extract the data
        time, field = self.simulation.diagnostics.get_moment_data(moment_name, time_idx)
        
        if plane == 'xy':
            field = PostProcessor.to_real_space(self.simulation,field)

        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot the data
        im = ax.imshow(field[:, :, z_idx].T, origin='lower', cmap='RdBu')
        if cbar: plt.colorbar(im, ax=ax, label=moment_name)
        if clim: im.set_clim(clim)
        
        ax.set_title(f'{moment_name} at t = {time:.2f}, amp: {np.max(np.abs(field[:, :, z_idx])):.2e}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Save the figure
        plt.tight_layout()
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/{moment_name}_t{time_idx}.png', dpi=150)
            plt.close(fig)
        if filename:
            fig.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()