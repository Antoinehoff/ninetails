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
        self.ikx0 = geometry.ikx0
        self.iky0 = geometry.iky0
        self.z = geometry.z
        self.params = geometry.params
        self.nkx = geometry.nkx
        self.nky = geometry.nky
        self.nz = geometry.nz
        self.dz = (self.z[-1] - self.z[0]) / self.nz if self.nz > 1 else 1

        self.K0 = geometry.K0
        self.K1 = geometry.K1
        self.K2 = geometry.K2

        # For quick access
        self.jacobian = geometry.jacobian
        
        sumKer2 = (self.K0**2 * self.K1**2 * self.K2**2)
        self.coeff = 1 + (1 - sumKer2) / self.params.tau

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
        if self.nz == 1:
            return 0
        else:
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
        (1 - 2[lperp - tau*lperp^2])phi = <phi>_yz + n + tau*lperp*(T_perp - n)
        """
        phi = y[-1]
        N00 = y[0]
        N01 = y[3]
        N02 = y[8]

        # Direct method to solve for phi
        y[-1] = self.flux_surf_avg(phi) + self.K0*N00 + self.K1*N01 + self.K2*N02
        y[-1] /= self.coeff
        y[-1, self.ikx0, self.iky0, :] = 0.0
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