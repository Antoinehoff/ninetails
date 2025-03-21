import numpy as np

from .config import SimulationConfig
from .diagnostics import Diagnostics
from .tools import get_grids
from .geometry import create_geometry
from .integrator import Integrator
from .models import HighOrderFluid
from .plotter import Plotter
from .boundary_conditions import BoundaryConditions

class Simulation:
    def __init__(self, input_file=None, config=None):
        '''
        Initialize the simulation from default or input YAML file.
        In details, sets up the grid, initializes the state vector, and sets up the solver.

        Parameters:
        -----------
        input_file : str [optional]
            Path to the input YAML file
        config : SimulationConfig [optional]
            Configuration object
        '''
        if input_file is None:
            self.config = SimulationConfig.create_default_config('input.yaml')
        elif config is not None:
            self.config = config
        else:
            self.config = SimulationConfig.from_yaml(input_file)
        self.setup()

    def run(self):
        '''Run the simulation'''
        self.RKscheme.integrate(self.equations.rhs, self.y0, self.t_span, self.config.numerical.dt, self.BC)
    
    def set_simulationconfig(self, **kwargs):
        '''
        Set parameters in the SimulationConfig object. Possible parameters are:
        - geometry_type, e.g. zpinch, s-alpha
        - nonlinear, boolean flag for nonlinear terms
        - model_type, e.g. HM, HW, MHW, 9GM
        - input_file, path to the input file
        - sim_name, name of the simulation
        - nframes, number of frames
        - output_dir, path to the output directory
        - restart, boolean flag for restart
        - restart_file, path to the restart file
        - save_restart, boolean flag for saving restart files
        - restart_interval, interval for saving restart files
        '''
        for key, value in kwargs.items():
            setattr(self.config, key, value)
        self.setup()

    def set_numericalconfig(self, **kwargs):
        '''
        Set numerical parameters for the simulation. Possible parameters are:
        - nx, number of grid points in x
        - ny, number of grid points in y
        - nz, number of grid points in z
        - Lx, domain size in x
        - Ly, domain size in y
        - Lz, domain size in z
        - dt, initial time step
        - max_time, maximum simulation time
        - muHD, hyperdiffusion coefficient
        '''
        for key, value in kwargs.items():
            setattr(self.config.numerical, key, value)
        self.setup()

    def set_physicalconfig(self, **kwargs):
        '''
        Set physical parameters for the simulation.
        Possible parameters are:
        - tau, ion-electron temperature ratio
        - RN, normalized density gradient for 9GM
        - RT, normalized temperature gradient for 9GM
        - eps, inverse aspect ratio for s-alpha
        - shear, magnetic shear for s-alpha
        - alpha_MHD, MHD alpha parameter for s-alpha
        - q0, safety factor for s-alpha
        - R0, major radius for s-alpha
        - alpha, adiabaticity parameter for Hasegawa-Wakatani
        - kappa, curvature parameter for Hasegawa-Wakatani
        '''
        for key, value in kwargs.items():
            setattr(self.config.physical, key, value)
        self.setup()

    def setup(self):
        ''' Set up the simulation once the config is loaded'''
        # Extract parameters from config
        num_params = self.config.numerical

        # Create geometry and grids
        gridDict, self.geometry = create_geometry(self.config)
        self.ndims = [num_params.nx, num_params.ny, num_params.nz]
        self.boxdim = [num_params.Lx, num_params.Ly, num_params.Lz]
        self.grids = [gridDict['x'], gridDict['y'], gridDict['z']]
        self.kgrids = [gridDict['kx'], gridDict['ky'], gridDict['z']]
        self.nkdims = [len(self.kgrids[0]), len(self.kgrids[1]), len(self.kgrids[2])]

        # Extended dimensions to include ghosts in the z direction for periodicity and twist-and-shift
        self.ngz = 4 if self.ndims[2] > 1 else 0
        self.ndims_ext = [num_params.nx, num_params.ny, num_params.nz + self.ngz]
        self.nkdims_ext = [len(self.kgrids[0]), len(self.kgrids[1]), len(self.kgrids[2]) + self.ngz]

        # Set up time span and time points for output
        self.t_span = (0, num_params.max_time)
        self.t_diag = np.linspace(0, num_params.max_time, self.config.nframes)  # output points
        self.t_diag = list(self.t_diag)
        self.dt_diag = self.t_diag[1] - self.t_diag[0]

        # Initialize state vector
        self.nmom = 9
        self.y0 = np.array([np.zeros(self.nkdims_ext, dtype=np.complex128) for _ in range(self.nmom+1)])

        # Add some random noise to break symmetry
        N_real =  0.5 * np.random.normal(size=self.ndims)
        for iz in range(self.ndims[2]):
            self.y0[0][:, :, iz] = np.fft.rfft2(N_real[:, :, iz])

        self.verbose = False

        self.equations = HighOrderFluid(self.config, self.geometry)
        self.diagnostics = Diagnostics(self.config)
        self.RKscheme = Integrator(method='RK4', diagnostic=self.diagnostics, verbose=self.verbose)
        self.plotter = Plotter(self)
        self.BC = BoundaryConditions(y=self.y0, model=self.geometry.zbc, nghosts=self.ngz) 

    def info(self):
        '''Print the parameters for verification'''
        self.config.info()