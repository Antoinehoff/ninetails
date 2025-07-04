import numpy as np

def modified_hasegawa_wakatani_rhs(model, t, y, dydt_out=None):
    """
    Compute the right-hand side of the modified Hasegawa-Wakatani equations.
    
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

    # First update phi based on the Poisson equation
    y[-1] = -y[1] / model.kperp2_pos
    y[-1][model.kperp2 == 0] = 0
    
    # Use pre-allocated arrays to avoid allocations
    model._temp1[:] = y[0] - model.poisson_solver.flux_surf_avg(y[0])
    model._temp2[:] = y[-1] - model.poisson_solver.flux_surf_avg(y[-1])
    n_nz = model._temp1
    phi_nz = model._temp2
    
    # Density equation
    dydt[0][:] = model.p.alpha * (phi_nz - n_nz)
    dydt[0] += -model.p.kappa * model.iky * y[-1]
    dydt[0] += -model.p.muHD * model.kperp2**2 * y[0]
    
    # Vorticity equation
    dydt[1][:] = model.p.alpha * (phi_nz - n_nz)
    dydt[1] += -model.p.muHD * model.kperp2**2 * y[1]
    
    # Compute the nonlinear terms using Poisson brackets if enabled
    if model.nonlinear:
        dydt[0] -= model.pb.compute(y[-1], y[0])
        dydt[1] -= model.pb.compute(y[-1], y[1])
    
    # Return the derivatives, including dphidt=0
    return dydt
