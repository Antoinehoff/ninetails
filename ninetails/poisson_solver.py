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
        
        sumKer2 = (self.K0**2 + self.K1**2 + self.K2**2)
        self.coeff = 1 + (1 - sumKer2) / self.params.tau

        if self.nz > 1:
            self.inv_jacobian_integral = 1.0/tools_integral(self.jacobian, self.dz, axis=-1)
        else:
            self.inv_jacobian_integral = 1.0
        
        # Pre-allocate work arrays to avoid allocations in solve()
        self._work_phi_avg = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        self._work_rhs = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
    def flux_surf_avg(self, phi, result=None):
        """
        Compute the flux surface average of phi according to the equation:
        <phi>_yz = (1/∫dz J_xyz) ∫dz J_xyz * phi(k_x, k_y=0, z, t)
        
        Parameters:
        -----------
        phi : ndarray
            Electrostatic potential in Fourier space (k_x, k_y, z)
        result : ndarray, optional
            Pre-allocated output array
            
        Returns:
        --------
        ndarray
            Flux surface averaged phi with the same shape as input phi
        """
        if self.nz == 1:
            return 0
        else:
            if result is None:
                result = self._work_phi_avg
            
            # Zero the result array
            result[:] = 0
            
            # Extract the k_y = 0 components and set them in result
            result[:, 0, :] = phi[:, 0, :]

            # Compute the integrand J_xyz * phi
            integrand = self.jacobian * result
            
            # Compute the flux surface average by integrating over z
            # and normalizing by the integral of the Jacobian
            integral = tools_integral(integrand, self.dz, axis=-1)

            # Compute the flux surface average
            result[:] = self.inv_jacobian_integral * integral

            return result
    
    def solve(self, y):
        """
        Solve the quasineutrality equation:
        (1 - 2[lperp - tau*lperp^2])phi = <phi>_yz + n + tau*lperp*(T_perp - n)
        """
        phi = y[-1]
        N00 = y[0]
        N01 = y[3]
        N02 = y[8]

        # Direct method to solve for phi (using pre-allocated work array)
        self.flux_surf_avg(phi, self._work_phi_avg)
        
        # Build RHS: <phi>_yz + K0*N00 + K1*N01 + K2*N02
        self._work_rhs[:] = self._work_phi_avg + self.K0*N00 + self.K1*N01 + self.K2*N02
        
        # Solve: phi = RHS / coeff
        y[-1][:] = self._work_rhs / self.coeff
        y[-1, self.ikx0, self.iky0, :] = 0.0
        
        return y