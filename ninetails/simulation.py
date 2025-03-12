import numpy as np

from .config import SimulationConfig
from .diagnostics import Diagnostics
from .tools import get_grids
from .geometry import create_geometry
from .integrator import Integrator
from .models import HighOrderFluid
from .postprocessor import PostProcessor
from .plotter import Plotter

class Simulation:
    def __init__(self, input_file=None):
        '''
        Initialize the simulation from default or input YAML file.
        In details, sets up the grid, initializes the state vector, and sets up the solver.
        
        Parameters:
        -----------
        input_file : str [optional]
            Path to the input YAML file
        '''
        if input_file is None:
            config = SimulationConfig.create_default_config('input.yaml')
        else:
            config = SimulationConfig.from_yaml(input_file)

        self.config = config

        # Extract parameters from config
        phys_params = config.physical
        num_params = config.numerical

        # Create geometry and grids
        gridDict, self.geometry = create_geometry(config)
        self.ndims = [num_params.nx, num_params.ny, num_params.nz]
        self.boxdim = [num_params.Lx, num_params.Ly, num_params.Lz]
        self.grids = [gridDict['x'], gridDict['y'], gridDict['z']]
        self.kgrids = [gridDict['kx'], gridDict['ky'], gridDict['z']]
        self.nkdims = [len(self.kgrids[0]), len(self.kgrids[1]), len(self.kgrids[2])]

        # Set up time span and time points for output
        self.t_span = (0, num_params.max_time)
        self.t_diag = np.linspace(0, num_params.max_time, config.nframes)  # output points
        self.t_diag = list(self.t_diag)
        self.dt_diag = self.t_diag[1] - self.t_diag[0]

        # Initialize state vector
        self.nmom = 9
        self.y0 = np.array([np.zeros(self.nkdims, dtype=np.complex128) for _ in range(self.nmom+1)])

        # Add some random noise to break symmetry
        N_real =  0.5 * np.random.normal(size=self.ndims)
        for iz in range(self.ndims[2]):
            self.y0[0][:, :, iz] = np.fft.rfft2(N_real[:, :, iz])

        self.verbose = False

        self.refresh()

    def run(self):
        '''Run the simulation'''
        self.RKscheme.integrate(self.equations.rhs, self.y0, self.t_span, self.config.numerical.dt)
    
    def set_max_time(self, new_time):
        '''Change the maximum time of the simulation'''
        self.config.numerical.max_time = new_time
        self.t_span = (0, new_time)
        self.t_diag = np.linspace(0, new_time, self.config.nframes)
        self.t_diag = list(self.t_diag)
        self.dt_diag = self.t_diag[1] - self.t_diag[0]
        self.refresh()

    def refresh(self):
        ''''''
        self.equations = HighOrderFluid(self.config, self.geometry)
        self.diagnostics = Diagnostics(self.config)
        self.RKscheme = Integrator(method='RK4', diagnostic=self.diagnostics, verbose=self.verbose)
        self.plotter = Plotter(self)

    def info(self):
        '''Print the parameters for verification'''
        self.config.info()