# equations.py
import numpy as np
from scipy.special import factorial
from .poisson_bracket import PoissonBracket
from .poisson_solver import PoissonSolver

class HighOrderFluid:
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
        
        self.l_perp = geometry.l_perp
        self.Cperp = geometry.Cperp  # Perpendicular curvature
        self.Cpar = geometry.Cpar    # Parallel curvature
        self.CparB = geometry.CparB  # Parallel magnetic curvature
        self.iky = 1j * ky_grid[:, :, np.newaxis]
        self.ikx = 1j * kx_grid[:, :, np.newaxis]
    
        # Set the right-hand side function based on the model type
        if self.model_type == 'GM9':
            self.rhs = self.GM9
        elif self.model_type == 'GM4':
            self.rhs = self.GM4
        elif self.model_type == 'HM':
            self.rhs = self.hasegawa_mima_rhs
        elif self.model_type == 'HW':
            self.rhs = self.hasegawa_wakatani_rhs
        elif self.model_type == 'MHW':
            self.rhs = self.modified_hasegawa_wakatani_rhs
        else:
            raise ValueError("Unknown solver type: {}".format(self.model_type))

    def hasegawa_mima_rhs(self, t, y):
        """
        Compute the right-hand side of the Hasegawa-Mima equations.
        """
        laplace_phi = -self.kperp2 * y[-1]

        y[-1] = -y[0] / (1.0 + self.kperp2)

        # Density equation
        self.dydt[0] = self.p.kappa * self.iky * y[-1] \
                     - self.p.alpha * self.ikx * y[-1] \
                     - self.p.muHD  * self.kperp2**2 * y[0]
        
        # Compute the nonlinear terms using Poisson brackets if enabled
        if self.nonlinear:
            self.dydt[0] += - self.pb.compute(y[-1], laplace_phi)
        # Return the derivatives, including dphidt=0
        return self.dydt

    def hasegawa_wakatani_rhs(self, t, y):
        """
        Compute the right-hand of the Hasegawa-Wakatani equations.
        """
        y[-1] = -y[1]/self.kperp2_pos
        y[-1][self.kperp2 == 0] = 0
        
        # Density equation
        self.dydt[0] = self.p.alpha * (y[-1] - y[0]) \
                     - self.p.kappa * self.iky * y[-1] \
                     - self.p.muHD * self.kperp2**2 * y[0]
        # Vorticity equation
        self.dydt[1] = self.p.alpha * (y[-1] - y[0]) \
                     - self.p.muHD * self.kperp2**2 * y[1]
        # Compute the nonlinear terms using Poisson brackets if enabled
        if self.nonlinear:
            self.dydt[0] -= self.pb.compute(y[-1], y[0])
            self.dydt[1] -= self.pb.compute(y[-1], y[1])
        
        # Return the derivatives, including dphidt=0
        return self.dydt
    
    def modified_hasegawa_wakatani_rhs(self, t, y):
        """
        Compute the right-hand side of the modified Hasegawa-Wakatani equations.
        """
        
        y[-1] = -y[1]/self.kperp2_pos
        y[-1][self.kperp2 == 0] = 0
        
        n_nz = y[0] - self.poisson_solver.flux_surf_avg(y[0])
        phi_nz = y[-1] - self.poisson_solver.flux_surf_avg(y[-1])
        
        # Density equation
        self.dydt[0] = self.p.alpha * (phi_nz - n_nz) \
                     - self.p.kappa * self.iky * y[-1] \
                     - self.p.muHD * self.kperp2**2 * y[0]
        
        # Vorticity equation
        self.dydt[1] = self.p.alpha * (phi_nz - n_nz) \
                     - self.p.muHD * self.kperp2**2 * y[1]
        
        # Compute the nonlinear terms using Poisson brackets if enabled
        if self.nonlinear:
            self.dydt[0] -= self.pb.compute(y[-1], y[0])
            self.dydt[1] -= self.pb.compute(y[-1], y[1])
        
        # Return the derivatives, including dphidt=0
        return self.dydt
    
    def GM4(self, t, y):
        """
        Compute the right-hand side of the fluid equations using the GM (2,1) equations.
        """
        # First update phi based on the Poisson equation
        y = self.poisson_solver.solve(y)
        
        # Unpack the moments
        N, u_par, T_par, T_perp, q_par, q_perp, _, _, _, phi = y
        tau = self.p.tau
        sqrt_tau = np.sqrt(tau)
        sqrt2 = np.sqrt(2)
        n_na = N + (1/tau - self.l_perp + 0.5*self.l_perp**2*tau) * phi
        Tperp_na = T_perp + (self.l_perp - self.l_perp**2*tau) * phi
        #'''
        # Add linear terms (always included)        
        # curvature terms
        self.dydt[0] = - self.Mperppj(0,0)
        '''
        # Thesis version (incomplete)
        self.dydt[0] = -tau * self.Cperp(sqrt2 * T_par - T_perp + 2*N) #
        self.dydt[0] -= sqrt_tau * (self.Cpar(u_par) - self.CparB(u_par)) #
        self.dydt[0] -= (2*self.Cperp(phi) + self.p.RN * self.iky * phi) #
        self.dydt[0] += tau * (3*self.Cperp(self.l_perp*phi) + (self.p.RT + self.p.RN) * self.iky * self.l_perp*phi) #
        
        self.dydt[1] = -sqrt_tau * self.Cpar(N) #
        self.dydt[1] -= 4.0 * tau * self.Cperp(u_par) #
        self.dydt[1] -= 6.0 * tau * self.Cperp(q_par) #
        self.dydt[1] += 1.0 * tau * self.Cperp(q_perp) #
        self.dydt[1] -= 2.0 * sqrt_tau * (self.Cpar(T_par) - self.CparB(T_par)) #
        self.dydt[1] += sqrt_tau * self.CparB(T_perp) #
        
        self.dydt[2] = -6.0 * tau * self.Cperp(T_par) #
        self.dydt[2] -= 2.0 * sqrt_tau * self.Cpar(u_par) #
        self.dydt[2] -= 0.5 * (1 - self.l_perp) * self.iky * self.p.RT * phi #
        
        self.dydt[3] = -4.0 * tau * self.Cperp(T_perp) #
        self.dydt[3] -= sqrt_tau * self.CparB(u_par) #
        self.dydt[3] -= (self.l_perp * self.iky * self.p.RN \
                       + (3 * self.l_perp - 1) * self.iky * self.p.RT) * phi #
        '''
        # Compute the nonlinear terms using Poisson brackets if enabled
        if self.nonlinear:
            # Prepare modified potentials for Poisson brackets
            phi_mod1 = (1 - self.l_perp) * phi
            phi_mod2 = self.l_perp * phi

            # Equation (A1): density
            self.dydt[0] -= self.pb.compute(phi_mod1, N)
            self.dydt[0] -= self.pb.compute(phi_mod2, T_perp)
            
            # Equation (A2): parallel velocity
            self.dydt[1] -= self.pb.compute(phi_mod1, u_par)
            
            # Equation (A3): parallel temperature
            self.dydt[2] -= self.pb.compute(phi_mod1, T_par)
            
            # Equation (A4): perpendicular temperature
            self.dydt[3] -= self.pb.compute(phi_mod1, T_perp)  # Corrected with phi_mod3
            self.dydt[3] += tau * self.pb.compute(phi_mod1, N)  # Corrected sign
        
        # Return the derivatives, including dphidt=0
        return self.dydt
    

    def GM9(self, t, y):
        """
        Compute the right-hand side of the fluid equations using the 9GM framework.
        """
        # First update phi based on the Poisson equation
        y = self.poisson_solver.solve(y)
        
        # Unpack the moments
        N, u_par, T_par, T_perp, q_par, q_perp, P_parpar, P_parperp, P_perpperp, phi = y
        tau = self.p.tau
        sqrt_tau = np.sqrt(tau)
        
        # Initialize the derivatives with zeros
        dN_dt = np.zeros_like(N)
        du_par_dt = np.zeros_like(u_par)
        dT_par_dt = np.zeros_like(T_par)
        dT_perp_dt = np.zeros_like(T_perp)
        dq_par_dt = np.zeros_like(q_par)
        dq_perp_dt = np.zeros_like(q_perp)
        dP_parpar_dt = np.zeros_like(P_parpar)
        dP_perppar_dt = np.zeros_like(P_parperp)
        dP_perpperp_dt = np.zeros_like(P_perpperp)

        # Add linear terms (always included)        
        # Equation (A1): density
        dN_dt = -2 * tau * self.Cperp(T_par - T_perp + N) #
        dN_dt -= sqrt_tau * (self.Cpar(u_par) - self.CparB(u_par)) #
        dN_dt -= ((1 - self.l_perp) * self.iky * self.p.RN - self.l_perp * self.iky * self.p.RT) * phi #
        
        # Equation (A2): parallel velocity
        du_par_dt = -sqrt_tau * self.Cpar(N) #
        du_par_dt -= 4.0 * tau * self.Cperp(u_par) #
        du_par_dt -= 6.0 * tau * self.Cperp(q_par) #
        du_par_dt += 1.0 * tau * self.Cperp(q_perp) #
        du_par_dt -= 2.0 * sqrt_tau * (self.Cpar(T_par) - self.CparB(T_par)) #
        du_par_dt += sqrt_tau * self.CparB(T_perp) #
        
        # Equation (A3): parallel temperature
        dT_par_dt = -6.0 * tau * self.Cperp(T_par) #
        dT_par_dt -= 2/3 * tau * self.Cperp(P_parpar) #
        dT_par_dt += 1.0 * tau * self.Cperp(P_parperp) #
        dT_par_dt -= 3.0 * sqrt_tau * (self.Cpar(q_par) - self.CparB(q_par)) #
        dT_par_dt += 2.0 * sqrt_tau * self.CparB(q_perp) #
        dT_par_dt -= 2.0 * sqrt_tau * self.Cpar(u_par) #
        dT_par_dt -= 0.5 * (1 - self.l_perp) * self.iky * self.p.RT * phi #
        
        # Equation (A4): perpendicular temperature
        dT_perp_dt = -4.0 * tau * self.Cperp(T_perp) #
        dT_perp_dt += 1.0 * tau * self.Cperp(N - 2 * P_parperp + 2 * P_perpperp) #
        dT_perp_dt -= sqrt_tau * (self.Cpar(q_perp) - 2 * self.CparB(q_perp)) #
        dT_perp_dt -= sqrt_tau * self.CparB(u_par) #
        dT_perp_dt -= (self.l_perp * self.iky * self.p.RN \
                       + (3 * self.l_perp - 1) * self.iky * self.p.RT) * phi #
        
        # Equation (A5): parallel heat flux
        dq_par_dt =  2.0 * sqrt_tau * (self.CparB(P_parpar) - self.Cpar(P_parpar)) #
        dq_par_dt += 3.0 * sqrt_tau * self.CparB(P_parperp) #
        dq_par_dt -= 3.0 * sqrt_tau * self.Cpar(T_par) #
        
        # Equation (A6): perpendicular heat flux
        dq_perp_dt = -2.0 * sqrt_tau * (self.Cpar(P_parperp) - 2 * self.CparB(P_parperp)) #
        dq_perp_dt += 2.0 * sqrt_tau * self.CparB(P_perpperp) #
        dq_perp_dt -= sqrt_tau * (self.Cpar(T_perp) + self.CparB(T_perp)) #
        dq_perp_dt -= 2.0 * sqrt_tau * self.CparB(T_par) #
        
        # Compute the nonlinear terms using Poisson brackets if enabled
        if self.nonlinear:
            # Prepare modified potentials for Poisson brackets
            phi_mod1 = (1 - self.l_perp) * phi
            phi_mod2 = self.l_perp * phi

            # Equation (A1): density
            dN_dt -= self.pb.compute(phi_mod1, N)
            dN_dt -= self.pb.compute(phi_mod2, T_perp)
            
            # Equation (A2): parallel velocity
            du_par_dt -= self.pb.compute(phi_mod1, u_par)
            du_par_dt -= self.pb.compute(phi_mod2, q_perp)
            
            # Equation (A3): parallel temperature
            dT_par_dt -= self.pb.compute(phi_mod1, T_par)
            dT_par_dt -= self.pb.compute(phi_mod2, P_parperp)
            
            # Equation (A4): perpendicular temperature
            dT_perp_dt -= self.pb.compute(phi_mod1, T_perp)  # Corrected with phi_mod3
            dT_perp_dt -= 0.5 * self.pb.compute(phi_mod2, P_perpperp)
            dT_perp_dt += tau * self.pb.compute(phi_mod1, N)  # Corrected sign
            
            # Equation (A5): parallel heat flux
            dq_par_dt -= self.pb.compute(phi, q_par)
            
            # Equation (A6): perpendicular heat flux
            dq_perp_dt -= self.pb.compute(phi, q_perp)
            dq_perp_dt += self.pb.compute(phi, u_par)
            
            # Equation (A7): parallel-parallel pressure tensor
            dP_parpar_dt -= self.pb.compute(phi, P_parpar)
            
            # Equation (A8): perpendicular-parallel pressure tensor
            dP_perppar_dt -= self.pb.compute(phi, P_parperp)
            dP_perppar_dt += self.pb.compute(phi, T_par)
            
            # Equation (A9): perpendicular-perpendicular pressure tensor
            dP_perpperp_dt -= self.pb.compute(phi, P_perpperp)
            dP_perpperp_dt += 0.5 * self.pb.compute(phi, T_perp)
            dP_perpperp_dt -= 0.25 * self.pb.compute(phi, N)
        
        # Return the derivatives, including dphidt=0
        return np.array([dN_dt, du_par_dt, dT_par_dt, dT_perp_dt, dq_par_dt, dq_perp_dt, 
                dP_parpar_dt, dP_perppar_dt, dP_perpperp_dt, np.zeros_like(phi)])
        
        
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