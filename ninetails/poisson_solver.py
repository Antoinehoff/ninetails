# poisson_solver.py
import numpy as np
from .tools import simpson_integral as tools_integral

class PoissonSolver:
    def __init__(self, geometry):
        """
        Initialize Poisson solver for the quasineutrality equation.
        
        Parameters:
        -----------
        geometry : Geometry
            Geometry object containing metric information and grid details
        """
        self.geometry = geometry
        self.kx = geometry.kx
        self.ky = geometry.ky
        self.z = geometry.z
        self.params = geometry.params
        self.nkx = geometry.nkx
        self.nky = geometry.nky
        self.nz = geometry.nz
        self.dz = (self.z[-1] - self.z[0]) / self.nz if self.nz > 1 else 1
        
        # For quick access
        self.jacobian = geometry.jacobian
        self.taulperp = self.params.tau * geometry.l_perp
        self.coeff = 1 - 2 * (geometry.l_perp - self.params.tau * geometry.l_perp**2)
        if self.nz > 1:
            self.inv_jacobian_integral = 1.0/tools_integral(self.jacobian, self.dz, axis=-1)
        else:
            self.inv_jacobian_integral = 1.0
        
    def flux_surf_avg(self, phi):
        """
        Compute the flux surface average of phi according to the equation:
        <phi>_yz = (1/∫dz J_xyz) ∫dz J_xyz * phi(k_x, k_y=0, z, t)
        
        Parameters:
        -----------
        phi : ndarray
            Electrostatic potential in Fourier space (k_x, k_y, z)
            
        Returns:
        --------
        ndarray
            Flux surface averaged phi with the same shape as input phi
        """
        # Extract the k_y = 0 components
        phi_ky0 = phi[:, 0, :]

        phi_avg = np.zeros_like(phi)
        phi_avg[:, 0, :] = phi_ky0

        # Compute the integrand J_xyz * phi
        integrand = self.jacobian * phi_avg
        
        # Compute the flux surface average by integrating over z
        # and normalizing by the integral of the Jacobian
        integral = tools_integral(integrand, self.dz, axis=-1)

        # Compute the flux surface average
        phi_avg = self.inv_jacobian_integral * integral

        return phi_avg
    
    def solve(self, y):
        """
        Solve the quasineutrality equation:
        (1 - 2[l_perp - tau*l_perp^2])phi = <phi>_yz + n + tau*l_perp*(T_perp - n)
        """

        # Direct method to solve for phi
        y[-1] = self.flux_surf_avg(y[-1]) + y[0] + self.taulperp * (y[3] - y[0])
        y[-1] /= self.coeff
        # Iterative method
        if False:
            # Initial guess for phi (ignoring flux surface average for now)
            phi = rhs / coeff
            
            # Now we need to account for the flux surface average term
            # We'll use an iterative approach to solve for phi
            max_iter = 20
            tol = 1e-10
            
            for i in range(max_iter):
                # Compute the flux surface average of the current phi
                phi_avg = self.flux_surf_avg(phi)
                
                # Update phi using the flux surface average
                phi_new = (rhs + phi_avg) / coeff
                
                # Check convergence
                phi_diff = np.max(np.abs(phi_new - phi))
                if phi_diff < tol:
                    break
                    
                phi = phi_new
        
        return y