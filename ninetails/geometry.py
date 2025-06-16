# geometry.py
import numpy as np

from .tools import get_grids

def create_geometry(config):
    gridDict = get_grids(config.numerical)

    if config.geometry_type.lower() == 'salpha':
        return gridDict, SAlphaGeometry(gridDict['kx'], gridDict['ky'], gridDict['z'], config.physical)
    elif config.geometry_type.lower() == 'zpinch':
        return gridDict, ZPinchGeometry(gridDict['kx'], gridDict['ky'], gridDict['z'], config.physical)
    else:
        raise ValueError(f"Unknown geometry type: {config.geometry_type}")

class Geometry:
    zbc = None # Boundary condition for z (to be implemented in subclasses)
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

        self.ikx0 = np.argmin(np.abs(kx))
        self.iky0 = np.argmin(np.abs(ky))
        
        # Initialize arrays
        self.gxx = np.zeros((self.nkx, self.nky, self.nz))
        self.gxy = np.zeros((self.nkx, self.nky, self.nz))
        self.gxz = np.zeros((self.nkx, self.nky, self.nz))
        self.gyy = np.zeros((self.nkx, self.nky, self.nz))
        self.gyz = np.zeros((self.nkx, self.nky, self.nz))
        self.g_zz = np.zeros((self.nkx, self.nky, self.nz))
        
        self.jacobian = np.zeros((self.nkx, self.nky, self.nz))
        self.hatB = np.zeros((self.nkx, self.nky, self.nz))
        self.dlnBdx = np.zeros((self.nkx, self.nky, self.nz))
        self.dlnBdy = np.zeros((self.nkx, self.nky, self.nz))
        self.dlnBdz = np.zeros((self.nkx, self.nky, self.nz))

        self.Cxy = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        self.Cperp = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        self.Cpar = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        self.CparB = np.zeros((self.nkx, self.nky, self.nz), dtype=np.complex128)
        
        # Compute metric coefficients
        self.compute_metrics()
        
        # Compute k_perp^2 and lperp
        self.compute_kperp_and_lperp()
        
        # Compute curvature operators
        self.get_curvature_operators()
    
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
        Compute k_perp^2 and lperp based on the metric coefficients
        """
        kx_2d, ky_2d = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        # Compute k_perp^2 according to equation (A12)
        kx_3d = kx_2d[:, :, np.newaxis]
        ky_3d = ky_2d[:, :, np.newaxis]
        
        self.kperp2 = (self.gxx * kx_3d**2 + 
                      2 * self.gxy * kx_3d * ky_3d + 
                      self.gyy * ky_3d**2)
        
        # Compute lperp as defined in equation (A11)
        self.lperp = 0.5 * self.params.tau * self.kperp2

        # Compute the first kernels
        # self.K0 = 1 - self.lperp + 0.5 * self.lperp**2
        # self.K1 = self.lperp - self.lperp**2
        # self.K2 = 0.5*self.lperp**2

        # Or get the analytic expression for the kernels
        self.K0 = np.exp(-self.lperp)
        self.K1 = self.lperp * np.exp(-self.lperp)
        self.K2 = 0.5 * self.lperp**2 * np.exp(-self.lperp)
    
    def get_curvature_operators(self):
        """
        Compute all curvature operators
        """

        for i in range(self.nkx):
            for j in range(self.nky):
                for k in range(self.nz):
                    i_kx = 1j * self.kx[i]
                    i_ky = 1j * self.ky[j]
                    gxx = self.gxx[i, j, k]
                    gxy = self.gxy[i, j, k]
                    gyz = self.gyz[i, j, k]
                    gyy = self.gyy[i, j, k]
                    gxy = self.gxy[i, j, k]
                    gxz = self.gxz[i, j, k]
                    dlnBdx = self.dlnBdx[i, j, k]
                    dlnBdy = self.dlnBdy[i, j, k]
                    dlnBdz = self.dlnBdz[i, j, k]

                    Gamma1 = gxx * gyy - gxy * gxy
                    Gamma2 = gxx * gyz - gxy * gxz
                    Gamma3 = gxy * gyz - gyy * gxz

                    G21 = Gamma2 / Gamma1
                    G31 = Gamma3 / Gamma1

                    self.Cxy[i, j, k] = - (dlnBdy + G21 * dlnBdz) * i_kx \
                                        + (dlnBdx - G31 * dlnBdz) * i_ky
                    
        # i_kx = np.zeros_like(self.gxx, dtype=np.complex128)
        # i_ky = np.zeros_like(self.gxx, dtype=np.complex128)

        # for i in range(self.nkx):
        #     for j in range(self.nky):
        #         for k in range(self.nz):
        #             i_kx[i, j, k] = 1j * self.kx[i]
        #             i_ky[i, j, k] = 1j * self.ky[j]

        # Gamma1 = self.gxx * self.gyy - self.gxy * self.gxy
        # Gamma2 = self.gxx * self.gyz - self.gxy * self.gxz
        # Gamma3 = self.gxy * self.gyz - self.gyy * self.gxz

        # G21 = Gamma2/Gamma1
        # G31 = Gamma3/Gamma1

        # self.Cxy =  -(self.dlnBdy + G21 * self.dlnBdz) * i_kx \
        #             +(self.dlnBdx - G31 * self.dlnBdz) * i_ky

        # g_xz = self.jacobian**2*(self.gxy * self.gyz - self.gyy * self.gxz)
        # g_yz = self.jacobian**2*(self.gxy * self.gxz - self.gxx * self.gyz)
        # g_zz = self.jacobian**2*(self.gxx * self.gyy - self.gxy**2)
        # Cx = -(self.dlnBdy - g_yz/g_zz * self.dlnBdz) / self.hatB
        # Cy =  (self.dlnBdx - g_xz/g_zz * self.dlnBdz) / self.hatB
        # self.Cxy = (Cx * i_kx + Cy * i_ky)


        self.Cperp = self.Cxy_op
        self.Cpar = self.Cz_op
        self.CparB = self.Cbz_op
    
    def Cxy_op(self,fin):
        """
        Compute the perpendicular curvature operator Cxy as defined in equation (A13)
        """
        return self.Cxy * fin
    
    def Cz_op(self, fin):
        """
        Compute the parallel curvature operator C∥f as defined in equation (A14)
        """
        return 0 #self.Cpar_factor * np.gradient(fin, axis=2)

    def Cbz_op(self, fin):
        """
        Compute the parallel magnetic curvature C∥^B as defined in equation (A15)
        """
        return self.Cz_op(self.hatB) * fin

class ZPinchGeometry(Geometry):
    zbc = 'periodic'
    def compute_metrics(self):
        # In Z-pinch, metrics are simpler (cartesian)
        R0 = getattr(self.params, 'R0', 1.0)
        # In Z-pinch, metrics are uniform in x,y at each z
        for iz in range(self.nz):
            self.gxx[:, :, iz] = 1.0
            self.gyy[:, :, iz] = 1.0
            self.g_zz[:, :, iz] = 1.0 ## 1.0/R0**2
            self.jacobian[:, :, iz] = 1.0 ## R0**2
            self.hatB[:, :, iz] = 1.0
            self.dlnBdx[:, :, iz] = -1.0
            self.dlnBdy[:, :, iz] = 0.0
            self.dlnBdz[:, :, iz] = 0.0

class SAlphaGeometry(Geometry):
    zbc = 'twist_and_shift'
    def compute_metrics(self):
        # Extract parameters
        shear = getattr(self.params, 'shear', 0.0)
        alpha_MHD = getattr(self.params, 'alpha_MHD', 0.0)
        eps = getattr(self.params, 'eps', 0.1)  # Default inverse aspect ratio
        q0 = getattr(self.params, 'q0', 2.0)    # Default safety factor
        
        # Compute metric components at each z location
        for iz, zz in enumerate(self.z):
            self.gxx[:, :, iz] = 1
            self.gyy[:, :, iz] = 1 + (shear * zz)**2
            self.gxy[:, :, iz] = shear * zz
            self.gyz[:, :, iz] = 1 / eps
            self.g_zz[:, :, iz] = 1 / eps**2

            hatB = 1 / (1 + eps * np.cos(zz))
            self.hatB[:, :, iz] = hatB
            self.jacobian[:, :, iz] = q0/hatB
            
            self.dlnBdx[:, :, iz] = -np.cos(zz) * hatB**2
            self.dlnBdy[:, :, iz] = 0
            self.dlnBdz[:, :, iz] = eps * np.sin(zz) * hatB**2