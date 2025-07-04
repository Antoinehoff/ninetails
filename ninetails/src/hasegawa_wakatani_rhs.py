import numpy as np

def hasegawa_wakatani_rhs(model, t, y, dydt_out=None):
    """
    Compute the right-hand side of the Hasegawa-Wakatani equations.
    
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

    y[-1] = -y[1] / model.kperp2_pos
    y[-1][model.kperp2 == 0] = 0
    
    # Density equation
    dydt[0][:] = model.p.alpha * (y[-1] - y[0])
    dydt[0] += -model.p.kappa * model.iky * y[-1]
    dydt[0] += -model.p.muHD * model.kperp2**2 * y[0]
    
    # Vorticity equation
    dydt[1][:] = model.p.alpha * (y[-1] - y[0])
    dydt[1] += -model.p.muHD * model.kperp2**2 * y[1]
    
    # Compute the nonlinear terms using Poisson brackets if enabled
    if model.nonlinear:
        dydt[0] -= model.pb.compute(y[-1], y[0])
        dydt[1] -= model.pb.compute(y[-1], y[1])
    
    # Return the derivatives, including dphidt=0
    return dydt
