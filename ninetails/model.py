# equations.py
import numpy as np
from .poisson_bracket import PoissonBracket
from .poisson_solver import PoissonSolver
from .src import GM9, GM4, hasegawa_mima_rhs, hasegawa_wakatani_rhs, modified_hasegawa_wakatani_rhs
from scipy.special import factorial

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
        
        self.dydt = np.array([np.zeros([self.nkx,self.nky,self.nz]) for i in range(10)],dtype=np.complex128)
        
        # Initialize zero array template
        kx_grid, ky_grid = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.zeros = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
        # Initialize Poisson solver
        self.poisson_solver = PoissonSolver(geometry)
        
        # Initialize Poisson bracket calculator
        self.pb = PoissonBracket(kx_grid, ky_grid)
        
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
        if self.model_type == 'GM9':
            self.rhs = lambda t, y: GM9(self, t, y)  # Pass self to GM9
        elif self.model_type == 'GM4':
            self.rhs = lambda t, y: GM4(self, t, y)  # Pass self to GM4
        elif self.model_type == 'HM':
            self.rhs = lambda t, y: hasegawa_mima_rhs(self, t, y)  # Pass self to hasegawa_mima_rhs
        elif self.model_type == 'HW':
            self.rhs = lambda t, y: hasegawa_wakatani_rhs(self, t, y)  # Pass self to hasegawa_wakatani_rhs
        elif self.model_type == 'MHW':
            self.rhs = lambda t, y: modified_hasegawa_wakatani_rhs(self, t, y)  # Pass self to modified_hasegawa_wakatani_rhs
        else:
            raise ValueError("Unknown solver type: {}".format(self.model_type))


    # Define the gyromoment hierarchy terms
    def Mna(self,p,j):
        # return the p,j non adiabatic moment
        nadiab = 1.0/self.p.tau * self.kernel(j)*(self.y[-1]) if p == 0 \
            else 0.0
        if (p,j) == (0,0):
            n = 0
        if (p,j) == (1,0):
            n = 1
        if (p,j) == (2,0):
            n = 2
        if (p,j) == (0,1):
            n = 3
        if (p,j) == (1,1):
            n = 4
        if (p,j) == (4,0):
            n = 5
        if (p,j) == (2,1):
            n = 6
        if (p,j) == (0,2):
            n = 7
        if (p,j) == (5,0):
            n = 8
        if (p,j) == (3,1):
            n = 9
        return self.y[n] + nadiab
    
    def Mparapj(self, p, j):
        curlyNpm1j = np.sqrt(p+1,j) * self.Mna(p+1,j) + np.sqrt(p) * self.Mna(p-1,j)
        curlyNpm1jm1 = np.sqrt(p+1,j-1) * self.Mna(p+1,j-1) + np.sqrt(p) * self.Mna(p-1,j-1)
        # Here Cpar = sqrt(tau)/sigma/Jacobian/hatB ddz
        return self.Cpar(curlyNpm1j) - self.CparB*((j+1) * curlyNpm1j - j * curlyNpm1jm1) \
            + self.CparB * np.sqrt(p) * \
                ((2*j+1) * self.Mna(p-1,j) - (j-1) * self.Mna(p-1,j+1) - j * self.Mna(p-1,j-1))
        
    def Mperppj(self, p, j):
        
        tau = self.p.tau
        q = self.p.q
        cpp2 = np.sqrt((p+1)*(p+2))
        cp   = 2*p + 1
        cpm2 = np.sqrt(p*(p-1))
        cjp1 = (2*j + 1)
        cj   = -(j+1)
        cjm1 = -j
                
        return tau/q * (
            self.Cperp(cpp2 * self.Mna(p+2,j) + cp * self.Mna(p,j) + cpm2 * self.Mna(p-2,j))
            + self.Cperp(cjp1 * self.Mna(p,j+1) + cj * self.Mna(p,j) + cjm1 * self.Mna(p,j-1))
        )
        
    def Dpj(self, p,j):
        
        if p == 0:
            return self.p.RN * self.kernel(j) + self.p.RT * ( -self.kernel(j) + 
                         ((2*j+1) * self.kernel(j) - (j+1) * self.kernel(j+1) - j*self.kernel(j-1))
                         ) * self.iky * self.phi
            
    def kernel(self, j):
        return self.l_perp**j * np.exp(-self.l_perp)/factorial(j)