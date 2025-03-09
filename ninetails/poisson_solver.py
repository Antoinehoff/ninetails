# poisson_solver.py
import numpy as np

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
        
        # For quick access
        self.kperp2 = geometry.kperp2
        self.l_perp = geometry.l_perp
        self.jacobian = geometry.jacobian
        
    def compute_flux_surface_average(self, phi):
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
        
        # Compute the integrand J_xyz * phi
        integrand = self.jacobian * phi_ky0
        
        # Compute the flux surface average by integrating over z
        # and normalizing by the integral of the Jacobian
        dz = (self.z[-1] - self.z[0]) / self.nz if self.nz > 1 else 1
        integral = np.sum(integrand, axis=1) * dz
        jacobian_integral = np.sum(self.jacobian) * dz
        
        # Compute the flux surface average
        phi_avg = integral / jacobian_integral
        
        # Create an array of the same shape as phi that contains the flux surface average
        # but only for k_y = 0 modes
        phi_avg_full = np.zeros_like(phi)
        phi_avg_full[:, 0, :] = phi_avg[:, np.newaxis]
        
        return phi_avg_full
    
    def solve(self, N, Tperp, phi):
        """
        Solve the quasineutrality equation:
        (1 - 2[l_perp - tau*l_perp^2])phi - <phi>_yz = n + tau*l_perp*(T_perp - n)
        
        Parameters:
        -----------
        N : ndarray
            Density fluctuation in Fourier space
        T_perp : ndarray
            Perpendicular temperature fluctuation in Fourier space
        phi : ndarray
            Electrostatic potential in Fourier space
            
        Returns:
        --------
        ndarray
            Electrostatic potential phi in Fourier space
        """
        # Compute the coefficient of phi on the LHS
        tau = self.params.tau
        coeff = 1 - 2 * (self.l_perp - tau * self.l_perp**2)
        
        # Compute the RHS: n + tau*l_perp*(T_perp - n)
        rhs = N + tau * self.l_perp * (Tperp - N)
        
        # Direct method to solve for phi
        phi_fs_avg = self.compute_flux_surface_average(phi)
        phi = (rhs + phi_fs_avg) / coeff
        
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
                phi_avg = self.compute_flux_surface_average(phi)
                
                # Update phi using the flux surface average
                phi_new = (rhs + phi_avg) / coeff
                
                # Check convergence
                phi_diff = np.max(np.abs(phi_new - phi))
                if phi_diff < tol:
                    break
                    
                phi = phi_new
        
        return phi