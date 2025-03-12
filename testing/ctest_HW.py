# main.py
import numpy as np
import os, sys
import ninetails as ntl

# Create a default configuration file if it doesn't exist
config = ntl.SimulationConfig.from_yaml('ctest_HW.yaml')

# Extract parameters from config
phys_params = config.physical
num_params = config.numerical

num_params.max_time = 50
config.nframes = 50

config.info()

# Set up the grid
nx, ny, nz = num_params.nx, num_params.ny, num_params.nz
Lx, Ly, Lz = num_params.Lx, num_params.Ly, num_params.Lz

grids = ntl.get_grids(config.numerical)
x = grids['x']
y = grids['y']
z = grids['z']
kx = grids['kx']
ky = grids['ky']

# Create geometry object first
geometry = ntl.create_geometry(kx, ky, z, phys_params, config.geometry_type)

# Set up time span and time points for output
t_span = (0, num_params.max_time)
t_diag = np.linspace(0, num_params.max_time, config.nframes)  # output points
t_diag = list(t_diag)
dt_diag = t_diag[1] - t_diag[0]

# Initialize diagnostics
diagnostics = ntl.Diagnostics(config)

# Create meshgrid for initialization
x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

# Initialize state vector
y0 = np.array([np.zeros((nx, ny//2+1, nz), dtype=np.complex128) for _ in range(9+1)])

# Add some random noise to break symmetry
N_real =  0.5 * np.random.normal(size=(nx, ny, nz))
for iz in range(nz):
    y0[0][:, :, iz] = np.fft.rfft2(N_real[:, :, iz])

# Initialize equation system with the geometry
equations = ntl.HighOrderFluid(config, geometry)

# Run the solver
RKscheme = ntl.Integrator(method='RK4', diagnostic=diagnostics, verbose=False)
RKscheme.integrate(equations.rhs, y0, t_span, config.numerical.dt)

# Post-process the results
postproc = ntl.PostProcessor(diagnostics)
postproc.compute_growth_rates()
postproc.plot_energy_evolution()
postproc.plot_2D_snapshot(moment_name='phi',time_idx=-1)
postproc.create_gif(moment_name='N', moviename=f'{config.output_dir}/ctest_HW_N.gif')
postproc.create_gif(moment_name='zeta', moviename=f'{config.output_dir}/ctest_HW_zeta.gif')
postproc.create_gif(moment_name='phi', moviename=f'{config.output_dir}/ctest_HW_phi.gif')