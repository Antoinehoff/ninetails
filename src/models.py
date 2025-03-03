# equations.py
import numpy as np
from poisson_bracket import PoissonBracket
from poisson_solver import PoissonSolver

class HighOrderFluid:
    def __init__(self, geometry, nonlinear=True):
        """
        Initialize the fluid equation system based on the gyro-moment framework.
        
        Parameters:
        -----------
        geometry : Geometry
            Geometry object containing metric information and grid details
        nonlinear : bool, optional
            Whether to include nonlinear terms
        """
        self.geometry = geometry
        self.p = geometry.params
        self.nonlinear = nonlinear
        
        # Extract grid parameters
        self.kx = geometry.kx
        self.ky = geometry.ky
        self.z = geometry.z
        self.nkx = geometry.nkx
        self.nky = geometry.nky
        self.nz = geometry.nz
        
        # Initialize zero array template
        kx_grid, ky_grid = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.zeros = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
        # Initialize Poisson solver
        self.poisson_solver = PoissonSolver(geometry)
        
        # Initialize Poisson bracket calculator
        self.pb = PoissonBracket(kx_grid, ky_grid)
        
        # For compact notation in the equations
        self.kperp2 = geometry.kperp2
        self.l_perp = geometry.l_perp
        self.Cxy = geometry.Cxy  # Perpendicular curvature
        self.Cz = geometry.Cz    # Parallel curvature
        self.CBz = geometry.CBz  # Parallel magnetic curvature
        self.iky = 1j * ky_grid[:, :, np.newaxis]
    
    def update_phi(self, y):
        """
        Update phi in the state vector using the Poisson equation
        
        Parameters:
        -----------
        y : list of ndarrays
            State vector containing moments and phi
            
        Returns:
        --------
        list of ndarrays
            Updated state vector with new phi
        """
        # Unpack the moments
        # N, u_par, T_par, T_perp, q_par, q_perp, P_parpar, P_perppar, P_perpperp, phi = y
        
        # Solve for phi
        new_phi = self.poisson_solver.solve(y[0], y[3], y[-1])
        
        # Replace phi in state vector
        y[9] = new_phi
        
        return y
    
    def rhs(self, t, y):
        """
        Compute the right-hand side of the fluid equations.
        
        Parameters:
        -----------
        t : float
            Current time
        y : list of ndarrays
            List containing [N, u_par, T_par, T_perp, q_par, q_perp, P_parpar, P_perppar, P_perpperp, phi]
            
        Returns:
        --------
        list of ndarrays
            Time derivatives of each moment (with dphidt=0)
        """
        # First update phi based on the Poisson equation
        y = self.update_phi(y)
        
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
        dphidt = np.zeros_like(phi)  # phi is determined by constraint, so dphidt=0
        
        # Prepare modified potentials for Poisson brackets
        phi_mod1 = (1 - self.l_perp) * phi
        phi_mod2 = self.l_perp * phi
        phi_mod3 = (1 - tau * self.l_perp) * phi  # For T_perp equation
        
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
            dT_perp_dt -= self.pb.compute(phi_mod3, T_perp)  # Corrected with phi_mod3
            dT_perp_dt -= 0.5 * self.pb.compute(phi_mod2, P_perpperp)
            dT_perp_dt -= tau * self.pb.compute(phi_mod1, N)  # Corrected sign
            
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
        dN_dt -= 2 * tau * self.Cxy * (T_par - T_perp + N)  # Corrected to use Cxy
        dN_dt -= (self.Cz - self.CBz) * sqrt_tau * u_par
        dN_dt -= ((1 - self.l_perp) * self.iky * self.p.RN - self.l_perp * self.iky * self.p.RT) * phi
        
        # Equation (A2): parallel velocity
        du_par_dt -= N * self.Cz * sqrt_tau  # Corrected to include N multiplier
        du_par_dt -= 4 * tau * self.Cxy * u_par  # Corrected to use Cxy
        du_par_dt -= 6 * tau * self.Cxy * q_par  # Corrected to use Cxy
        du_par_dt += tau * self.Cxy * q_perp  # Corrected to use Cxy and sign
        du_par_dt -= 2 * (self.Cz - self.CBz) * sqrt_tau * T_par
        du_par_dt -= self.CBz * sqrt_tau * T_perp
        
        # Equation (A3): parallel temperature
        dT_par_dt -= 6 * tau * self.Cxy * T_par  # Corrected to use Cxy
        dT_par_dt -= (2/3) * tau * self.Cxy * P_parpar  # Corrected to use Cxy and sign
        dT_par_dt += tau * self.Cxy * P_perppar  # Corrected to use Cxy and sign
        dT_par_dt -= 3 * sqrt_tau * (self.Cz - self.CBz) * q_par
        dT_par_dt += 2 * sqrt_tau * self.CBz * q_perp  # Corrected sign
        dT_par_dt -= 2 * self.Cz * sqrt_tau * u_par
        dT_par_dt -= ((1 - self.l_perp) / 2) * self.iky * self.p.RT * phi
        
        # Equation (A4): perpendicular temperature
        dT_perp_dt -= 4 * tau * self.Cxy * T_perp  # Corrected to use Cxy
        dT_perp_dt += tau * self.Cxy * (N - 2 * P_perppar + 2 * P_perpperp)  # Corrected to use Cxy and sign
        dT_perp_dt += sqrt_tau * (self.Cz - 2 * self.CBz) * q_perp  # Corrected sign
        dT_perp_dt += self.CBz * sqrt_tau * u_par
        dT_perp_dt -= (self.l_perp * self.iky * self.p.RN + (3 * self.l_perp - 1) * self.iky * self.p.RT) * phi
        
        # Equation (A5): parallel heat flux
        dq_par_dt -= self.pb.compute(phi, q_par)
        dq_par_dt += 2 * sqrt_tau * (self.CBz - self.Cz) * P_parpar
        dq_par_dt += 3 * sqrt_tau * self.CBz * P_perppar
        dq_par_dt -= 3 * self.Cz * sqrt_tau * T_par
        
        # Equation (A6): perpendicular heat flux
        dq_perp_dt -= self.pb.compute(phi, q_perp)
        dq_perp_dt += self.pb.compute(phi, u_par)
        dq_perp_dt += 2 * sqrt_tau * (self.Cz - 2 * self.CBz) * P_perppar  # Corrected sign
        dq_perp_dt += 2 * sqrt_tau * self.CBz * P_perpperp
        dq_perp_dt -= sqrt_tau * (self.Cz + self.CBz) * T_perp
        dq_perp_dt -= 2 * self.CBz * sqrt_tau * T_par
        
        # Return the derivatives, including dphidt=0
        return [dN_dt, du_par_dt, dT_par_dt, dT_perp_dt, dq_par_dt, dq_perp_dt, 
                dP_parpar_dt, dP_perppar_dt, dP_perpperp_dt, dphidt]