import numpy as np

def get_grids(numparam):
        # Set up the spatial grid in real space
    x = np.linspace(0, numparam.Lx, numparam.nx, endpoint=False)
    y = np.linspace(0, numparam.Ly, numparam.ny, endpoint=False)
    
    # Set up the wavenumber grid in Fourier space
    kx = 2*np.pi*np.fft.fftfreq(numparam.nx, numparam.Lx/numparam.nx)
    ky = 2*np.pi*np.fft.rfftfreq(numparam.ny, numparam.Ly/numparam.ny)

    # Define z between -pi and pi (or user-defined range)
    z = np.linspace(-np.pi, np.pi, numparam.nz, endpoint=False) if numparam.nz > 1 else np.array([0])
    return {
        'x': x,
        'y': y,
        'z': z,
        'kx': kx,
        'ky': ky
    }

def dfdz_o4(fxyz, nz, dz):
    """
    Compute the 4th order finite difference of a function f
    
    Parameters:
    -----------
    f : array_like
        Function to differentiate. Must have two ghost points on each side
        The ghost are placed at the end of the array, so the two lower ghosts are
            f[-2], f[-1]
        The two upper ghosts are
            f[n], f[n+1]
        where n is the number of grid points.

    dx : float
        Grid spacing
    
    Returns:
    --------
    dfdx : array_like
        4th order finite difference of f
    """
    # Compute the derivative according to (1/12 f-2 - 2/3 f-1 + 2/3 f+1 - 1/12 f+2)/dz
    for iz in range(nz):
        dfdz = (1/12*fxyz[:,:,iz-2] - 2/3*fxyz[:,:,iz-1] + 2/3*fxyz[:,:,iz+1] - 1/12*fxyz[:,:,iz+2])/dz
    return dfdz
