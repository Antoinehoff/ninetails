# main.py
import numpy as np
import os
from scipy.integrate import solve_ivp
from models import HighOrderFluid
from geometry import create_geometry
from config import SimulationConfig
from diagnostics import Diagnostics
from post_processing import PostProcessor

def main():
    # Load configuration from YAML file
    config_file = "simulation_config.yaml"
    if not os.path.exists(config_file):
        # Create a default configuration file if it doesn't exist
        create_default_config(config_file)
    
    config = SimulationConfig.from_yaml(config_file)
    
    # Extract parameters from config
    phys_params = config.physical
    num_params = config.numerical
    
    # Set up the grid
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    Lx, Ly, Lz = num_params.Lx, num_params.Ly, num_params.Lz
    
    # Create output directory if it doesn't exist
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # Set up the spatial grid in real space
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    
    # Set up the wavenumber grid in Fourier space
    kx = np.fft.fftfreq(nx, Lx/nx)
    ky = np.fft.rfftfreq(ny, Ly/ny)
    
    # Define z between -pi and pi (or user-defined range)
    z = np.linspace(-np.pi, np.pi, nz, endpoint=False) if nz > 1 else np.array([0])
    
    # Create geometry object first
    geometry = create_geometry(kx, ky, z, phys_params, config.geometry_type)
    
    # Initialize equation system with the geometry
    equations = HighOrderFluid(geometry, nonlinear=config.nonlinear)
    
    # Set up time span and time points for output
    t_span = (0, num_params.max_time)
    t_eval = np.linspace(0, num_params.max_time, 100)  # 100 output points
    
    # Initialize diagnostics
    grid = {
        'x': x,
        'y': y,
        'z': z,
        'kx': kx,
        'ky': ky
    }
    diagnostics = Diagnostics(grid, phys_params)
    
    # Initialize moments in real space with appropriate perturbations
    # Gaussian perturbation in density with random phases
    sigma_x = Lx / 10  # Width of Gaussian in x
    sigma_y = Ly / 10  # Width of Gaussian in y
    amplitude = 1e-4   # Small perturbation amplitude
    
    # Create meshgrid for initialization
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    
    # Initialize all moments in real space
    N_real = np.zeros((nx, ny, nz))
    u_par_real = np.zeros((nx, ny, nz))
    T_par_real = np.zeros((nx, ny, nz))
    T_perp_real = np.zeros((nx, ny, nz))
    q_par_real = np.zeros((nx, ny, nz))
    q_perp_real = np.zeros((nx, ny, nz))
    P_parpar_real = np.zeros((nx, ny, nz))
    P_perppar_real = np.zeros((nx, ny, nz))
    P_perpperp_real = np.zeros((nx, ny, nz))
    
    # Create a localized perturbation in density
    # for iz in range(nz):
    #     # Gaussian profile centered in the domain
    #     N_real[:, :, iz] = amplitude * np.exp(
    #         -((x_grid - Lx/2)**2 / (2 * sigma_x**2) + 
    #           (y_grid - Ly/2)**2 / (2 * sigma_y**2))
    #     )
    
    # Add some random noise to break symmetry
    N_real = amplitude * 0.1 * np.random.normal(size=(nx, ny, nz))
    
    # Transform to Fourier space
    N_hat = np.zeros((nx, ny//2+1, nz), dtype=np.complex128)
    u_par_hat = np.zeros_like(N_hat)
    T_par_hat = np.zeros_like(N_hat)
    T_perp_hat = np.zeros_like(N_hat)
    q_par_hat = np.zeros_like(N_hat)
    q_perp_hat = np.zeros_like(N_hat)
    P_parpar_hat = np.zeros_like(N_hat)
    P_perppar_hat = np.zeros_like(N_hat)
    P_perpperp_hat = np.zeros_like(N_hat)
    phi_hat = np.zeros_like(N_hat)
    
    # Perform 2D FFT for each z plane
    for iz in range(nz):
        N_hat[:, :, iz] = np.fft.rfft2(N_real[:, :, iz])
        u_par_hat[:, :, iz] = np.fft.rfft2(u_par_real[:, :, iz])
        T_par_hat[:, :, iz] = np.fft.rfft2(T_par_real[:, :, iz])
        T_perp_hat[:, :, iz] = np.fft.rfft2(T_perp_real[:, :, iz])
        q_par_hat[:, :, iz] = np.fft.rfft2(q_par_real[:, :, iz])
        q_perp_hat[:, :, iz] = np.fft.rfft2(q_perp_real[:, :, iz])
        P_parpar_hat[:, :, iz] = np.fft.rfft2(P_parpar_real[:, :, iz])
        P_perppar_hat[:, :, iz] = np.fft.rfft2(P_perppar_real[:, :, iz])
        P_perpperp_hat[:, :, iz] = np.fft.rfft2(P_perpperp_real[:, :, iz])
    
    # Initialize phi by solving the Poisson equation
    phi_hat = equations.poisson_solver.solve(N_hat, T_perp_hat, phi_hat)
    
    # Initial state vector including phi
    y0 = [
        N_hat, 
        u_par_hat, 
        T_par_hat, 
        T_perp_hat, 
        q_par_hat, 
        q_perp_hat, 
        P_parpar_hat, 
        P_perppar_hat, 
        P_perpperp_hat,
        phi_hat
    ]
    
    # Flatten y0 for the ODE solver
    y0_flat = np.concatenate([y.flatten() for y in y0])
    
    def rhs_wrapper(t, y_flat):
        """
        Wrapper for the RHS function that handles reshaping the flattened array
        and updates diagnostics at specified time points.
        
        Parameters:
        -----------
        t : float
            Current time
        y_flat : ndarray
            Flattened state vector
            
        Returns:
        --------
        ndarray
            Flattened derivatives
        """
        # Reshape the flattened array back into the moment arrays
        shapes = [y.shape for y in y0]
        sizes = [np.prod(shape) for shape in shapes]
        
        y_reshaped = []
        start_idx = 0
        for shape, size in zip(shapes, sizes):
            end_idx = start_idx + size
            y_reshaped.append(y_flat[start_idx:end_idx].reshape(shape))
            start_idx = end_idx
        
        # Compute the derivatives
        dydt = equations.rhs(t, y_reshaped)
        
        # Update diagnostics only at requested output times to avoid excessive updates
        # Find if current time is close to any evaluation time
        if diagnostics and any(abs(t - eval_t) < 1e-6 for eval_t in t_eval):
            diagnostics.update(t, y_reshaped[:-1], y_reshaped[-1])  # Moments and phi
        
        # Flatten the derivatives
        dydt_flat = np.concatenate([dy.flatten() for dy in dydt])
        return dydt_flat

    # Run the solver with adaptive timestepping
    print("Starting simulation...")
    solution = solve_ivp(
        rhs_wrapper, 
        t_span, 
        y0_flat, 
        method='RK45',
        t_eval=t_eval,
        atol=1e-8,
        rtol=1e-6
    )
    
    # Check solver status
    if solution.success:
        print("Solver completed successfully.")
    else:
        print("Solver encountered an issue:", solution.message)
        return
    
    # Reshape the solution for post-processing
    reshaped_solution = {}
    reshaped_solution['t'] = solution.t
    
    shapes = [y.shape for y in y0]
    sizes = [np.prod(shape) for shape in shapes]
    indices = np.cumsum(sizes)
    
    # Extract moments and phi from the solution
    moment_names = ['N', 'u_par', 'T_par', 'T_perp', 'q_par', 'q_perp', 
                   'P_parpar', 'P_perppar', 'P_perpperp', 'phi']
    
    reshaped_solution['moments'] = {}
    start_idx = 0
    
    for i, (name, shape, size) in enumerate(zip(moment_names, shapes, sizes)):
        end_idx = start_idx + size
        data = solution.y[start_idx:end_idx, :]
        reshaped_solution['moments'][name] = data.reshape(shape + (-1,))  # Add time dimension
        start_idx = end_idx
    
    # Create PostProcessor object
    postprocessor = PostProcessor(reshaped_solution, nx, ny, nz, x, y, z)
    
    # Generate standard plots
    print("Generating plots...")
    postprocessor.plot_energy_evolution(diagnostics.energy_history)
    postprocessor.plot_enstrophy_evolution(diagnostics.enstrophy_history)
    
    # Plot 2D snapshots at different times
    snapshot_indices = [0, len(t_eval) // 2, -1]
    for idx in snapshot_indices:
        postprocessor.plot_2D_snapshot('N', time_idx=idx)
        postprocessor.plot_2D_snapshot('phi', time_idx=idx)
    
    # Plot time evolution of selected modes
    postprocessor.plot_time_evolution('N', kx_idx=1, ky_idx=1)
    postprocessor.plot_time_evolution('phi', kx_idx=1, ky_idx=1)
    
    # Plot phi and its flux surface average
    postprocessor.plot_phi_and_avg(indices=snapshot_indices)

    print("Analyzing linear growth rates...")
    
    # Compute growth rates for density fluctuations
    postprocessor.compute_growth_rates(moment_name='N')
    
    # Compute growth rates for potential
    postprocessor.compute_growth_rates(moment_name='phi')
    
    # If you want theoretical predictions for comparison
    postprocessor.compute_theoretical_growth_rates(geometry_type=config.geometry_type)
    
    # You can also add specialized analysis for specific modes or parameter scans
    # For example, to track the evolution of a particular mode:
    key_mode_kx, key_mode_ky = 1, 1  # Example mode to track
    postprocessor.plot_time_evolution('N', kx_idx=key_mode_kx, ky_idx=key_mode_ky)
    
    print("Growth rate analysis completed.")
    
    print(f"Simulation completed. Results saved to {config.output_dir}/")

def create_default_config(filename):
    """Create a default configuration file"""
    config_data = """
# Default configuration for plasma fluid simulation
physical:
  tau: 0.01   # Temperature ratio (T_e/T_i)
  RN: 1.0     # Density gradient scale length
  RT: 100.0   # Temperature gradient scale length
  eps: 0.1    # Inverse aspect ratio (for s-alpha geometry)
  shear: 0.0  # Magnetic shear (for s-alpha geometry)
  alpha_MHD: 0.0  # MHD alpha parameter (for s-alpha geometry)
  q0: 2.0     # Safety factor (for s-alpha geometry)
  R0: 1.0     # Major radius (for s-alpha geometry)
  
numerical:
  nx: 64       # Number of grid points in x
  ny: 64       # Number of grid points in y
  nz: 1        # Number of grid points in z
  Lx: 100.0    # Domain size in x
  Ly: 100.0    # Domain size in y
  Lz: 2.0      # Domain size in z (only used if nz > 1)
  dt: 0.01     # Initial time step (for fixed-step methods)
  max_time: 100.0  # Maximum simulation time
  hyperdiffusion: 0.1  # Hyperdiffusion coefficient
  
geometry_type: 'zpinch'  # 'salpha' or 'zpinch'
nonlinear: true         # Include nonlinear terms
output_dir: 'output'    # Directory for output files
"""
    with open(filename, 'w') as f:
        f.write(config_data)
    print(f"Created default configuration file: {filename}")

if __name__ == "__main__":
    main()