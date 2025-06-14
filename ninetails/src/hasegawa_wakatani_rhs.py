import numpy as np

def hasegawa_wakatani_rhs(model, t, y):
    """
    Compute the right-hand side of the Hasegawa-Wakatani equations.
    """
    y[-1] = -y[1] / model.kperp2_pos
    y[-1][model.kperp2 == 0] = 0
    
    # Density equation
    model.dydt[0] = model.p.alpha * (y[-1] - y[0]) \
                  - model.p.kappa * model.iky * y[-1] \
                  - model.p.muHD * model.kperp2**2 * y[0]
    # Vorticity equation
    model.dydt[1] = model.p.alpha * (y[-1] - y[0]) \
                  - model.p.muHD * model.kperp2**2 * y[1]
    # Compute the nonlinear terms using Poisson brackets if enabled
    if model.nonlinear:
        model.dydt[0] -= model.pb.compute(y[-1], y[0])
        model.dydt[1] -= model.pb.compute(y[-1], y[1])
    
    # Return the derivatives, including dphidt=0
    return model.dydt
