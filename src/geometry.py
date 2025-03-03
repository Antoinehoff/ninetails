# geometry.py
import numpy as np

class Geometry:
    def __init__(self, kx, ky, z, params):
        """
        Base geometry class for the fluid model.
        
        Parameters:
        -----------
        kx : ndarray
            Radial wavenumbers (1D array)
        ky : ndarray
            Binormal wavenumbers (1D array)
        z : ndarray
            Parallel coordinate values (1D array)
        params : object
            Physical parameters
        """
        self.kx = kx
        self.ky = ky
        self.z = z
        self.params = params
        self.nkx = len(kx)
        self.nky = len(ky)
        self.nz = len(z)
        
        # Initialize arrays
        self.Cxy = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        self.Cz = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        self.CBz = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
        # Compute metric coefficients
        self.metrics = self.compute_metrics()
        
        # Compute k_perp^2 and l_perp
        self.compute_kperp_and_lperp()
        
        # Compute curvature operators
        self.compute_curvature_operators()
    
    def compute_metrics(self):
        """
        Compute the metric coefficients for the specific geometry.
        Must be implemented in subclasses.
        
        Returns:
        --------
        dict
            Dictionary containing the metric tensor components
        """
        raise NotImplementedError("Must be implemented in subclass")
    
    def compute_kperp_and_lperp(self):
        """
        Compute k_perp^2 and l_perp based on the metric coefficients
        """
        kx_2d, ky_2d = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        # Initialize for 3D arrays
        g_xx_3d = np.zeros((self.nkx, self.nky, self.nz))
        g_xy_3d = np.zeros((self.nkx, self.nky, self.nz))
        g_yy_3d = np.zeros((self.nkx, self.nky, self.nz))
        
        # Expand 2D metrics to 3D for all z
        for iz in range(self.nz):
            g_xx_3d[:, :, iz] = self.metrics['g_xx']
            g_xy_3d[:, :, iz] = self.metrics['g_xy']
            g_yy_3d[:, :, iz] = self.metrics['g_yy']
        
        # Store the 3D metric tensors
        self.g_xx = g_xx_3d
        self.g_xy = g_xy_3d
        self.g_yy = g_yy_3d
        
        # Compute k_perp^2 according to equation (A12)
        kx_3d = kx_2d[:, :, np.newaxis]
        ky_3d = ky_2d[:, :, np.newaxis]
        
        self.kperp2 = (self.g_xx * kx_3d**2 + 
                      2 * self.g_xy * kx_3d * ky_3d + 
                      self.g_yy * ky_3d**2)
        
        # Compute l_perp as defined in equation (A11)
        self.l_perp = 0.5 * self.params.tau * self.kperp2
    
    def compute_curvature_operators(self):
        """
        Compute all curvature operators
        """
        self.Cxy = self.compute_Cxy()
        self.Cz = self.compute_Cz()
        self.CBz = self.compute_CBz()
    
    def compute_Cxy(self):
        """
        Compute the perpendicular curvature operator Cxy as defined in equation (A13)
        
        Returns:
        --------
        ndarray
            Cxy operator array
        """
        raise NotImplementedError("Must be implemented in subclass")
    
    def compute_Cz(self):
        """
        Compute the parallel curvature operator C∥ as defined in equation (A14)
        
        Returns:
        --------
        ndarray
            C∥ operator array
        """
        raise NotImplementedError("Must be implemented in subclass")
    
    def compute_CBz(self):
        """
        Compute the parallel magnetic curvature C∥^B as defined in equation (A15)
        
        Returns:
        --------
        ndarray
            C∥^B operator array
        """
        raise NotImplementedError("Must be implemented in subclass")

class SAlphaGeometry(Geometry):
    def compute_metrics(self):
        """
        Compute metric coefficients for s-alpha geometry
        
        Returns:
        --------
        dict
            Dictionary containing the metric tensor components
        """
        # Extract parameters
        shear = getattr(self.params, 'shear', 0.0)
        alpha_MHD = getattr(self.params, 'alpha_MHD', 0.0)
        eps = getattr(self.params, 'eps', 0.1)  # Default inverse aspect ratio
        q0 = getattr(self.params, 'q0', 2.0)    # Default safety factor
        
        # Initialize metric tensors
        g_xx = np.ones((self.nkx, self.nky))
        g_xy = np.zeros((self.nkx, self.nky))
        g_yy = np.ones((self.nkx, self.nky))
        
        # Initialize Jacobian and B field
        self.jacobian = np.zeros(self.nz)
        self.hatB = np.zeros(self.nz)
        self.dBdx = np.zeros(self.nz)
        self.dBdy = np.zeros(self.nz)
        self.dBdz = np.zeros(self.nz)
        
        # Compute metric components at each z location
        for iz, zz in enumerate(self.z):
            # Compute sheared metric components
            g_xy_z = shear * zz - alpha_MHD * np.sin(zz)
            g_yy_z = 1 + (shear * zz - alpha_MHD * np.sin(zz))**2
            
            # Jacobian and B field
            self.jacobian[iz] = q0 * (1 + eps * np.cos(zz))
            self.hatB[iz] = 1 / (1 + eps * np.cos(zz))
            self.dBdz[iz] = eps * np.sin(zz) * self.hatB[iz]**2
            
            # For s-alpha geometry, the metrics are uniform in x,y at each z
            for ikx in range(self.nkx):
                for iky in range(self.nky):
                    g_xy[ikx, iky] = g_xy_z
                    g_yy[ikx, iky] = g_yy_z
        
        # Save and return metric dictionary
        metrics = {
            'g_xx': g_xx,
            'g_xy': g_xy,
            'g_yy': g_yy
        }
        
        return metrics
    
    def compute_Cxy(self):
        """
        Compute perpendicular curvature operator for s-alpha geometry
        according to equation (A13)
        
        Returns:
        --------
        ndarray
            Cxy operator
        """
        Cxy = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
        # Get the 2D meshgrid of kx, ky
        kx_2d, ky_2d = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        # Extract parameters
        eps = getattr(self.params, 'eps', 0.1)
        
        for iz, zz in enumerate(self.z):
            # Compute curvature components
            Cx = 0.0  # No x-component in s-alpha
            Cy = -eps * np.cos(zz)  # Standard s-alpha curvature
            
            # Apply to all k modes at this z location
            for ikx in range(self.nkx):
                for iky in range(self.nky):
                    Cxy[ikx, iky, iz] = Cx * kx_2d[ikx, iky] + Cy * ky_2d[ikx, iky]
        
        return Cxy
    
    def compute_Cz(self):
        """
        Compute parallel curvature operator for s-alpha geometry
        according to equation (A14)
        
        Returns:
        --------
        ndarray
            C∥ operator
        """
        Cz = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
        # Extract parameter
        R0 = getattr(self.params, 'R0', 1.0)
        
        for iz in range(self.nz):
            # Compute C∥ = (R0 / (Jxyz * B̂)) * ∂/∂z
            factor = R0 / (self.jacobian[iz] * self.hatB[iz])
            
            # Apply to all k modes
            Cz[:, :, iz] = factor
        
        return Cz
    
    def compute_CBz(self):
        """
        Compute parallel magnetic curvature for s-alpha geometry
        according to equation (A15): C∥^B = C∥ ln B
        
        Returns:
        --------
        ndarray
            C∥^B operator
        """
        CBz = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
        for iz in range(self.nz):
            # C∥^B = C∥ ln B
            lnB = np.log(self.hatB[iz])
            
            # Apply to all k modes
            CBz[:, :, iz] = self.Cz[:, :, iz] * lnB
        
        return CBz

class ZPinchGeometry(Geometry):
    def compute_metrics(self):
        """
        Compute metric coefficients for Z-pinch geometry
        
        Returns:
        --------
        dict
            Dictionary containing the metric tensor components
        """
        # In Z-pinch, metrics are simpler (cartesian)
        g_xx = np.ones((self.nkx, self.nky))
        g_xy = np.zeros((self.nkx, self.nky))
        g_yy = np.ones((self.nkx, self.nky))
        
        # Compute Jacobian and B field
        self.jacobian = np.ones(self.nz)
        self.hatB = np.ones(self.nz)
        self.dBdx = np.ones(self.nz)  # In Z-pinch, dB/dx = B
        self.dBdy = np.zeros(self.nz)
        self.dBdz = np.zeros(self.nz)
        
        # Return metrics dictionary
        return {
            'g_xx': g_xx,
            'g_xy': g_xy,
            'g_yy': g_yy
        }
    
    def compute_Cxy(self):
        """
        Compute perpendicular curvature operator for Z-pinch geometry
        according to equation (A13)
        
        Returns:
        --------
        ndarray
            Cxy operator
        """
        Cxy = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
        # Get the 2D meshgrid of kx, ky
        kx_2d, ky_2d = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        for iz in range(self.nz):
            # In Z-pinch, curvature is in y-direction
            Cx = 0.0
            Cy = 1.0  # Normalized curvature in Z-pinch
            
            # Apply to all k modes at this z location
            for ikx in range(self.nkx):
                for iky in range(self.nky):
                    Cxy[ikx, iky, iz] = Cx * kx_2d[ikx, iky] + Cy * ky_2d[ikx, iky]
        
        return Cxy
    
    def compute_Cz(self):
        """
        Compute parallel curvature operator for Z-pinch geometry
        according to equation (A14)
        
        Returns:
        --------
        ndarray
            C∥ operator
        """
        # In Z-pinch, this is a constant
        return np.ones((self.nkx, self.nky, self.nz), dtype=np.complex128)
    
    def compute_CBz(self):
        """
        Compute parallel magnetic curvature for Z-pinch geometry
        according to equation (A15): C∥^B = C∥ ln B
        
        Returns:
        --------
        ndarray
            C∥^B operator
        """
        # In Z-pinch, B is uniform in z, so C∥^B = 0
        return np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)

def create_geometry(kx, ky, z, params, geometry_type='salpha'):
    """
    Factory function to create appropriate geometry object
    
    Parameters:
    -----------
    kx : ndarray
        Radial wavenumbers (1D array)
    ky : ndarray
        Binormal wavenumbers (1D array)
    z : ndarray
        Parallel coordinate values (1D array)
    params : object
        Physical parameters
    geometry_type : str, optional
        Type of geometry ('salpha' or 'zpinch')
        
    Returns:
    --------
    Geometry
        Geometry object of appropriate type
    """
    if geometry_type.lower() == 'salpha':
        return SAlphaGeometry(kx, ky, z, params)
    elif geometry_type.lower() == 'zpinch':
        return ZPinchGeometry(kx, ky, z, params)
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")