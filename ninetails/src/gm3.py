import numpy as np

def zero_func(x): return 0

def GM3(model, t, y, dydt_out=None):
    """
    Compute the right-hand side of the fluid equations using the 9GM framework.
    
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

    # Access attributes from the model instance
    y = model.poisson_solver.solve(y)

    # Unpack the moments
    N00, _, n20, N01, _, _, _, _, _, phi = y
    tau = model.p.tau
    sqrt2 = np.sqrt(2)
    
    # Use pre-allocated temporary arrays to avoid allocations
    # n00 = N00 + model.K0 / tau * phi
    model._temp_n00[:] = N00 + model.K0 / tau * phi
    # n01 = N01 + model.K1 / tau * phi  
    model._temp_n01[:] = N01 + model.K1 / tau * phi
    
    n00 = model._temp_n00
    n01 = model._temp_n01

    # Add linear terms (always included)        
    # Equation (A1): density (p=0, j=0)
    model._temp1[:] = 2*n00 + sqrt2*n20 - n01
    dydt[0][:] = -tau * model.Cperp(model._temp1)
    dydt[0] += model.K1 * model.p.RT * model.iky * phi 
    dydt[0] += -model.K0 * model.p.RN * model.iky * phi
    model._temp1[:] = 5*n00 + sqrt2 * n20
    dydt[0] += -2./3. * model.p.nu * model.lperp * model._temp1
    dydt[0] += -model.p.muHD * model.kperp2 * y[0]

    # Equation (A2): parallel velocity (p=1, j=0)
    dydt[1][:] = 0  # Parallel velocity is not used in GM3
    
    # Equation (A3): parallel temperature (p=2, j=0)
    model._temp1[:] = 6*n20 + sqrt2*n00
    dydt[2][:] = tau * model.Cperp(model._temp1)
    dydt[2] += -0.5 * sqrt2 * model.p.RT * model.iky * model.K0 * phi
    model._temp1[:] = 2*sqrt2*n00 + 2*n20 + sqrt2*n01
    dydt[2] += -2./3. * model.p.nu * model._temp1
    dydt[2] += -model.p.muHD * model.kperp2 * y[2]

    # Equation (A4): perpendicular temperature (p=0, j=1)
    model._temp1[:] = 4*n01 - n00
    dydt[3][:] = tau * model.Cperp(model._temp1)
    model._temp1[:] = 2*model.K1 - model.K0 - 2*model.K2
    dydt[3] += -model.p.RT * model.iky * model._temp1 * phi
    dydt[3] += -model.p.RN * model.iky * model.K1 * phi
    model._temp1[:] = 2*n00 + sqrt2*n20 + n01
    dydt[3] += -2./3. * model.p.nu * model._temp1
    dydt[3] += -model.p.muHD * model.kperp2 * y[3]

    # Nonlinear terms
    if model.nonlinear:
        # Prepare modified potentials for Poisson brackets
        model._temp_K0phi[:] = model.K0 * phi
        model._temp_K1phi[:] = model.K1 * phi

        # Equation (A1): density
        dydt[0] -= model.pb.compute(model._temp_K0phi, N00)
                
        # Equation (A3): parallel temperature
        dydt[2] -= model.pb.compute(model._temp_K0phi, n20)
        
        # Equation (A4): perpendicular temperature
        dydt[3] -= model.pb.compute(model._temp_K1phi, N01)
    
    # Zero unused equations
    for i in [4, 5, 6, 7, 8]:
        dydt[i][:] = 0
    
    # Return the derivatives, including dphidt=0
    return dydt
