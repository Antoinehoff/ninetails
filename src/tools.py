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