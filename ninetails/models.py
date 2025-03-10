# equations.py
import numpy as np
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
    
    def rhs(self, t, y):
        """
        Compute the right-hand side of the fluid equations.
        """
        if self.model_type == '9GM':
            return self.gyro_moment_rhs(t, y)
        elif self.model_type == 'HW':
            return self.hasegawa_wakatani_rhs(t, y)
        elif self.model_type == 'MHW':
            return self.modified_hasegawa_wakatani_rhs(t, y)
        else:
            raise ValueError("Unknown solver type: {}".format(self.model_type))

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
        # Unpack the moments
        n = y[0]
        zeta = y[1]
        
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
    
    def gyro_moment_rhs(self, t, y):
        """
        Compute the right-hand side of the fluid equations using the 9GM framework.
        """
        # First update phi based on the Poisson equation
        y = self.poisson_solver.solve(y)
        
        # Unpack the moments
        N, u_par, T_par, T_perp, q_par, q_perp, P_parpar, P_perppar, P_perpperp, phi = y
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
        dP_perppar_dt = np.zeros_like(P_perppar)
        dP_perpperp_dt = np.zeros_like(P_perpperp)
        
        # Prepare modified potentials for Poisson brackets
        phi_mod1 = (1 - self.l_perp) * phi
        phi_mod2 = self.l_perp * phi
        
        # Compute the nonlinear terms using Poisson brackets if enabled
        if self.nonlinear:
            # Equation (A1): density
            dN_dt -= self.pb.compute(phi_mod1, N)
            dN_dt -= self.pb.compute(phi_mod2, T_perp)
            
            # Equation (A2): parallel velocity
            du_par_dt -= self.pb.compute(phi_mod1, u_par)
            du_par_dt -= self.pb.compute(phi_mod2, q_perp)
            
            # Equation (A3): parallel temperature
            dT_par_dt -= self.pb.compute(phi_mod1, T_par)
            dT_par_dt -= self.pb.compute(phi_mod2, P_perppar)
            
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
            dP_perppar_dt -= self.pb.compute(phi, P_perppar)
            dP_perppar_dt += self.pb.compute(phi, T_par)
            
            # Equation (A9): perpendicular-perpendicular pressure tensor
            dP_perpperp_dt -= self.pb.compute(phi, P_perpperp)
            dP_perpperp_dt += 0.5 * self.pb.compute(phi, T_perp)
            dP_perpperp_dt -= 0.25 * self.pb.compute(phi, N)
        
        # Add linear terms (always included)
        
        # Equation (A1): density
        dN_dt -= 2 * tau * self.Cperp * (T_par - T_perp + N)
        dN_dt -= (self.Cpar - self.CparB) * sqrt_tau * u_par
        dN_dt -= ((1 - self.l_perp) * self.iky * self.p.RN - self.l_perp * self.iky * self.p.RT) * phi
        
        # Equation (A2): parallel velocity
        du_par_dt -= N * self.Cpar * sqrt_tau
        du_par_dt -= 4 * tau * self.Cperp * u_par
        du_par_dt -= 6 * tau * self.Cperp * q_par
        du_par_dt += tau * self.Cperp * q_perp
        du_par_dt -= 2 * (self.Cpar - self.CparB) * sqrt_tau * T_par
        du_par_dt += self.CparB * sqrt_tau * T_perp
        
        # Equation (A3): parallel temperature
        dT_par_dt -= 6 * tau * self.Cperp * T_par
        dT_par_dt -= (2/3) * tau * self.Cperp * P_parpar
        dT_par_dt += tau * self.Cperp * P_perppar
        dT_par_dt -= 3 * sqrt_tau * (self.Cpar - self.CparB) * q_par
        dT_par_dt += 2 * sqrt_tau * self.CparB * q_perp
        dT_par_dt -= 2 * self.Cpar * sqrt_tau * u_par
        dT_par_dt -= ((1 - self.l_perp) / 2) * self.iky * self.p.RT * phi
        
        # Equation (A4): perpendicular temperature
        dT_perp_dt -= 4 * tau * self.Cperp * T_perp
        dT_perp_dt += tau * self.Cperp * (N - 2 * P_perppar + 2 * P_perpperp)
        dT_perp_dt -= sqrt_tau * (self.Cpar - 2 * self.CparB) * q_perp 
        dT_perp_dt -= self.CparB * sqrt_tau * u_par
        dT_perp_dt -= (self.l_perp * self.iky * self.p.RN + (3 * self.l_perp - 1) * self.iky * self.p.RT) * phi
        
        # Equation (A5): parallel heat flux
        dq_par_dt -= self.pb.compute(phi, q_par)
        dq_par_dt += 2 * sqrt_tau * (self.CparB - self.Cpar) * P_parpar
        dq_par_dt += 3 * sqrt_tau * self.CparB * P_perppar
        dq_par_dt -= 3 * self.Cpar * sqrt_tau * T_par
        
        # Equation (A6): perpendicular heat flux
        dq_perp_dt -= self.pb.compute(phi, q_perp)
        dq_perp_dt += self.pb.compute(phi, u_par)
        dq_perp_dt -= 2 * sqrt_tau * (self.Cpar - 2 * self.CparB) * P_perppar
        dq_perp_dt += 2 * sqrt_tau * self.CparB * P_perpperp
        dq_perp_dt -= sqrt_tau * (self.Cpar + self.CparB) * T_perp
        dq_perp_dt -= 2 * self.CparB * sqrt_tau * T_par
        
        # Return the derivatives, including dphidt=0
        return np.array([dN_dt, du_par_dt, dT_par_dt, dT_perp_dt, dq_par_dt, dq_perp_dt, 
                dP_parpar_dt, dP_perppar_dt, dP_perpperp_dt, np.zeros_like(phi)])
    