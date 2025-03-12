import sys
import numpy as np
from .file_utils import load_solution_hdf5
from .postprocessor import PostProcessor
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def analyze(output_file, option='all'):
    # Load the solution, diagnostics, and configuration from the HDF5 file
    diagnostics, config = load_solution_hdf5(output_file)
    diagnostics.config = config
    
    # Extract parameters from config
    phys_params = config.physical
    num_params = config.numerical
    
    # Set up the grid
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    Lx, Ly, Lz = num_params.Lx, num_params.Ly, num_params.Lz
    nt = diagnostics.frames['t'].size
    
    # Set up the spatial grid in real space
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    
    # Define z between -pi and pi (or user-defined range)
    z = np.linspace(-np.pi, np.pi, nz, endpoint=False) if nz > 1 else np.array([0])
        
    # Create PostProcessor object
    postprocessor = PostProcessor(diagnostics)
    
    if option == 'all' or option == 'energy':
        # Generate energy evolution plot
        print("Generating energy evolution plot...")
        postprocessor.plot_energy_evolution(diagnostics.integrated)
    
    if option == 'all' or option == 'enstrophy':
        # Generate enstrophy evolution plot
        print("Generating enstrophy evolution plot...")
        postprocessor.plot_enstrophy_evolution(diagnostics.enstrophy_history)
    
    if option == 'all' or option == 'snapshots':
        # Plot 2D snapshots at different times
        print("Generating 2D snapshots...")
        snapshot_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
        for idx in snapshot_indices:
            postprocessor.plot_2D_snapshot('N', time_idx=idx)
            postprocessor.plot_2D_snapshot('phi', time_idx=idx)
    
    if option == 'all' or option == 'time_evolution':
        # Plot time evolution of selected modes
        print("Generating time evolution plots...")
        postprocessor.plot_time_evolution('N', kx_idx=0, ky_idx=1)
        postprocessor.plot_time_evolution('phi', kx_idx=0, ky_idx=1)
    
    if option == 'all' or option == 'phi_avg':
        # Plot phi and its flux surface average
        print("Generating phi and flux surface average plots...")
        snapshot_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
        postprocessor.plot_phi_and_avg(indices=snapshot_indices)

    if option == 'all' or option == 'growth_rates':
        print("Analyzing linear growth rates...")
        # Compute growth rates for density fluctuations
        postprocessor.compute_growth_rates(moment_name='N')
        # Compute growth rates for potential
        postprocessor.compute_growth_rates(moment_name='phi')
        # If you want theoretical predictions for comparison
        postprocessor.compute_theoretical_growth_rates(geometry_type=config.geometry_type)
        # You can also add specialized analysis for specific modes or parameter scans
        # For example, to track the evolution of a particular mode:
        key_mode_kx, key_mode_ky = 0, 1  # Example mode to track
        postprocessor.plot_time_evolution('N', kx_idx=key_mode_kx, ky_idx=key_mode_ky)
    
    print(f"Analysis completed. Results saved to {config.output_dir}/")

if __name__ == "__main__":
    options = ['all', 'energy', 'enstrophy', 'snapshots', 'time_evolution', 'phi_avg', 'growth_rates']
    def errmsg():
        print("Usage: python analyze.py <output_file> [option]")
        print("Options: ", options)
        sys.exit(1)
        
    if len(sys.argv) < 2 or sys.argv[1] == '-h':
        errmsg()
    else:
        output_file = sys.argv[1]
        
        option = sys.argv[2] if len(sys.argv) > 2 else 'all'
        if option not in options: errmsg()
        
        analyze(output_file, option)