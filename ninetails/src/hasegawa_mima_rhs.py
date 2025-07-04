import numpy as np

def hasegawa_mima_rhs(model, t, y, dydt_out=None):
    """
    Compute the right-hand side of the Hasegawa-Mima equations.
    
    Parameters:
    -----------
    model : Model object
        Contains physics parameters and operators
    t : float
        Current time
    y : list of ndarrays
        Current state vector
    dydt_out : list of ndarrays, optional
        Pre-allocated output arrays. If None, uses model.dydt
    """
    # Use pre-allocated output arrays if provided
    if dydt_out is not None:
        dydt = dydt_out
    else:
        dydt = model.dydt

    # Use pre-allocated array for laplace_phi to avoid allocation
    model._temp1[:] = -model.kperp2 * y[-1]
    laplace_phi = model._temp1

    y[-1] = -y[0] / (1.0 + model.kperp2)

    # Density equation
    dydt[0][:] = model.p.kappa * model.iky * y[-1]
    dydt[0] += -model.p.alpha * model.ikx * y[-1]
    dydt[0] += -model.p.muHD * model.kperp2**2 * y[0]
    
    # Compute the nonlinear terms using Poisson brackets if enabled
    if model.nonlinear:
        dydt[0] += -model.pb.compute(y[-1], laplace_phi)
    
    # Return the derivatives, including dphidt=0
    return dydt
