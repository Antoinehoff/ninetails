# equations.py
import numpy as np
from .poisson_bracket import PoissonBracket
from .poisson_solver import PoissonSolver
from .src import GMX, GM9, GM4, GM3, \
    hasegawa_mima_rhs, hasegawa_wakatani_rhs, \
    modified_hasegawa_wakatani_rhs

class Model:
    def __init__(self, config, geometry):
        """
        Initialize the fluid equation system based on the 9GM framework.
        
        Parameters:
        -----------
        config : SimulationConfig
            Configuration object containing all parameters
        geometry : Geometry
            Geometry object containing metric information and grid details
        """
        self.geometry = geometry
        self.p = config.physical
        self.p.muHD = config.numerical.muHD
        self.nonlinear = config.nonlinear
        self.model_type = config.model_type
        
        # Extract grid parameters
        self.kx = geometry.kx
        self.ky = geometry.ky
        self.z = geometry.z
        self.nkx = geometry.nkx
        self.nky = geometry.nky
        self.nz = geometry.nz
        self.lperp = geometry.lperp
        
        self.dydt = np.array([np.zeros([self.nkx,self.nky,self.nz]) for i in range(10)],dtype=np.complex128)
        
        # Initialize zero array template
        kx_grid, ky_grid = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.zeros = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
        # Initialize Poisson solver
        self.poisson_solver = PoissonSolver(geometry)
        
        # Initialize Poisson bracket calculator
        if self.nonlinear:
            self.pb = PoissonBracket(kx_grid, ky_grid)
        else:
            zero_array = np.zeros((2,2))
            self.pb = PoissonBracket(zero_array, zero_array)  # Dummy PoissonBracket for linear case
            self.pb.compute = lambda f, g: 0  # Override compute method to return zero

        # For compact notation in the equations
        self.kperp2 = geometry.kperp2
        
        self.kperp2_pos = self.kperp2.copy()
        self.kperp2_pos[self.kperp2_pos == 0] = 1  # Avoid division by zero for poisson in HW
        
        self.K0 = geometry.K0
        self.K1 = geometry.K1
        self.K2 = geometry.K2
        self.lperp = geometry.lperp
        self.kperp2 = geometry.kperp2
        self.Cperp = geometry.Cperp  # Perpendicular curvature
        self.Cpar = geometry.Cpar    # Parallel curvature
        self.CparB = geometry.CparB  # Parallel magnetic curvature
        self.iky = 1j * ky_grid[:, :, np.newaxis]
        self.ikx = 1j * kx_grid[:, :, np.newaxis]
    
        # Set the right-hand side function based on the model type
        if self.model_type == 'GMX':
            self.rhs = lambda t, y: GMX(self, t, y)
        elif self.model_type == 'GM9':
            self.rhs = lambda t, y: GM9(self, t, y)  # Pass self to GM9
        elif self.model_type == 'GM4':
            self.rhs = lambda t, y: GM4(self, t, y)  # Pass self to GM4
        elif self.model_type == 'GM3':
            self.rhs = lambda t, y: GM3(self, t, y)
        elif self.model_type == 'HM':
            self.rhs = lambda t, y: hasegawa_mima_rhs(self, t, y)  # Pass self to hasegawa_mima_rhs
        elif self.model_type == 'HW':
            self.rhs = lambda t, y: hasegawa_wakatani_rhs(self, t, y)  # Pass self to hasegawa_wakatani_rhs
        elif self.model_type == 'MHW':
            self.rhs = lambda t, y: modified_hasegawa_wakatani_rhs(self, t, y)  # Pass self to modified_hasegawa_wakatani_rhs
        else:
            raise ValueError("Unknown solver type: {}".format(self.model_type))