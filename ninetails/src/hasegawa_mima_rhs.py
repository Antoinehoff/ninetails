import numpy as np

def hasegawa_mima_rhs(model, t, y):
    """
    Compute the right-hand side of the Hasegawa-Mima equations.
    """
    laplace_phi = -model.kperp2 * y[-1]

    y[-1] = -y[0] / (1.0 + model.kperp2)

    # Density equation
    model.dydt[0] = model.p.kappa * model.iky * y[-1] \
                  - model.p.alpha * model.ikx * y[-1] \
                  - model.p.muHD * model.kperp2**2 * y[0]
    
    # Compute the nonlinear terms using Poisson brackets if enabled
    if model.nonlinear:
        model.dydt[0] += -model.pb.compute(y[-1], laplace_phi)
    # Return the derivatives, including dphidt=0
    return model.dydt
