import numpy as np

def zero_func(x): return 0

def GM3(model, t, y):
    """
    Compute the right-hand side of the fluid equations using the 9GM framework.
    """

    # Access attributes from the model instance
    y = model.poisson_solver.solve(y)
    
    # Unpack the moments
    N00, _, n20, N01, _, _, _, _, _, phi = y
    tau = model.p.tau
    sqrt2 = np.sqrt(2)
    n00 = N00 + model.K0 / tau * phi
    n01 = N01 + model.K1 / tau * phi

    # Add linear terms (always included)        
    # Equation (A1): density (p=0, j=0)
    model.dydt[0]  = -tau * model.Cperp(2*n00 + sqrt2*n20 - n01) 
    model.dydt[0] += +model.K1 * model.p.RT * model.iky * phi 
    model.dydt[0] += -model.K0 * model.p.RN * model.iky * phi
    model.dydt[0] += -2./3. * model.p.nu * model.lperp * (5*n00 + sqrt2 * n20)
    model.dydt[0] += -model.p.muHD * model.kperp2 * y[0]
    
    # Equation (A3): parallel temperature (p=2, j=0)
    model.dydt[2]  = +tau * model.Cperp(6*n20 + sqrt2*n00)
    model.dydt[2] += -0.5 * sqrt2 * model.p.RT * model.iky * model.K0 * phi
    model.dydt[2] += -2./3. * model.p.nu * (2*sqrt2*n00 + 2*n20 + sqrt2*n01)
    model.dydt[2] += -model.p.muHD * model.kperp2 * y[2]

    # Equation (A4): perpendicular temperature (p=0, j=1)
    model.dydt[3]  = +tau * model.Cperp(4*n01 - n00)
    model.dydt[3] += -model.p.RT * model.iky * (2*model.K1 - model.K0 - 2*model.K2) * phi
    model.dydt[3] += -model.p.RN * model.iky * model.K1 * phi
    model.dydt[3] += -2./3. * model.p.nu * (2*n00 + sqrt2*n20 + n01)
    model.dydt[3] += -model.p.muHD * model.kperp2 * y[3]

    # Compute the nonlinear terms using Poisson brackets if enabled
    if model.nonlinear:
        # Prepare modified potentials for Poisson brackets
        K0phi = model.K0 * phi
        K1phi = model.K1 * phi

        # Equation (A1): density
        model.dydt[0] -= model.pb.compute(K0phi, N00)
        #model.dydt[0] -= model.pb.compute(K1phi, N01)
                
        # Equation (A3): parallel temperature
        model.dydt[2] -= model.pb.compute(K0phi, n20)
        
        # Equation (A4): perpendicular temperature
        model.dydt[3] -= model.pb.compute(K1phi, N01)
        #self.dydt[3] -= tau * self.pb.compute(K1phi, N00)
    
    # Return the derivatives, including dphidt=0
    return model.dydt
