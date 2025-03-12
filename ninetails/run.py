# main.py
import numpy as np
import os, sys
from .models import HighOrderFluid
from .geometry import create_geometry
from .config import SimulationConfig
from .diagnostics import Diagnostics
from .file_utils import save_solution_hdf5
from .tools import get_grids
from .integrator import Integrator

def run(config_file):
    if not os.path.exists(config_file):
        # Create a default configuration file if it doesn't exist
        SimulationConfig.create_default_config(config_file)
    
    config = SimulationConfig.from_yaml(config_file)
    
    # Extract parameters from config
    phys_params = config.physical
    num_params = config.numerical
    
    # Print the parameters for verification
    print(phys_params)
    print(num_params)
    
    # Set up the grid
    nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
    Lx, Ly, Lz = num_params.Lx, num_params.Ly, num_params.Lz
    
    # Create output directory if it doesn't exist
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    grids = get_grids(config.numerical)
    x = grids['x']
    y = grids['y']
    z = grids['z']
    kx = grids['kx']
    ky = grids['ky']

    # Create geometry object first
    geometry = create_geometry(kx, ky, z, phys_params, config.geometry_type)
    
    # Initialize equation system with the geometry
    equations = HighOrderFluid(config, geometry)
    
    # Set up time span and time points for output
    t_span = (0, num_params.max_time)
    t_diag = np.linspace(0, num_params.max_time, config.nframes)  # output points
    t_diag = list(t_diag)
    dt_diag = t_diag[1] - t_diag[0]
    
    # Initialize diagnostics
    diagnostics = Diagnostics(config)
    
    # Initialize moments in real space with appropriate perturbations
    # Gaussian perturbation in density with random phases
    sigma_x = 2.0  # Width of Gaussian in x
    sigma_y = 2.0  # Width of Gaussian in y
    amplitude = 0.5   # Small perturbation amplitude
    
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
    y0 = np.array([
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
    ])
    
    def rhs_wrapper(t, y):
        """
        Wrapper for the RHS function that handles reshaping the flattened array
        and updates diagnostics at specified time points.
        
        Parameters:
        -----------
        t : float
            Current time
        y : ndarray
            state vector
            
        Returns:
        --------
        ndarray
            derivatives
        """

        # Compute the derivatives
        dydt = equations.rhs(t, y)
        
        # Update diagnostics only at requested output times to avoid excessive updates
        # Find if current time is close to any evaluation time
        if diagnostics and len(t_diag) > 0:
            if abs(t - t_diag[0]) < dt_diag/10:
                diagnostics.update(t, y)
                # diagnostics.plot_on_the_fly()
                # remove all t_eval times that are less than or equal to current time
                t_diag.pop(0)
                print(f"t = {t:.2e}, Etot = {diagnostics.integrated['total'][-1]:.4e}")
    
        return dydt

    # Define the output interval
    dt_out = num_params.max_time / config.nframes  # Adjust as needed

    # Run the solver
    integrator = Integrator(method='RK4', nprint=100)
    integrator.integrate(rhs_wrapper, y0, t_span, dt_out)
    
if __name__ == "__main__":
    # config file is the first command line argument
    if len(sys.argv) < 2:
        print("Usage: python run.py <config_file>")
        sys.exit(1)
    if os.path.exists(sys.argv[1]):
        run(sys.argv[1])
    else:
        print("Config file not found:", sys.argv[1])
        sys.exit(1)