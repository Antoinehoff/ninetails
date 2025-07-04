import numpy as np

def GM9(model, t, y, dydt_out=None):
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
    N, u_par, T_par, T_perp, q_par, q_perp, P_parpar, P_parperp, P_perpperp, phi = y
    tau = model.p.tau
    sqrt_tau = np.sqrt(tau)
    
    # Use pre-allocated arrays instead of allocating new ones
    dN_dt = dydt[0]
    du_par_dt = dydt[1]
    dT_par_dt = dydt[2]
    dT_perp_dt = dydt[3]
    dq_par_dt = dydt[4]
    dq_perp_dt = dydt[5]
    dP_parpar_dt = dydt[6]
    dP_perppar_dt = dydt[7]
    dP_perpperp_dt = dydt[8]

    # Add linear terms (always included)
    # Equation (A1): density
    model._temp1[:] = T_par - T_perp + N
    dN_dt[:] = -2 * tau * model.Cperp(model._temp1)
    model._temp1[:] = model.Cpar(u_par) - model.CparB(u_par)
    dN_dt -= sqrt_tau * model._temp1
    dN_dt -= ((1 - model.lperp) * model.iky * model.p.RN - model.lperp * model.iky * model.p.RT) * phi
    
    # Equation (A2): parallel velocity
    du_par_dt[:] = -sqrt_tau * model.Cpar(N)
    du_par_dt -= 4.0 * tau * model.Cperp(u_par)
    du_par_dt -= 6.0 * tau * model.Cperp(q_par)
    du_par_dt += 1.0 * tau * model.Cperp(q_perp)
    model._temp1[:] = model.Cpar(T_par) - model.CparB(T_par)
    du_par_dt -= 2.0 * sqrt_tau * model._temp1
    du_par_dt += sqrt_tau * model.CparB(T_perp)
    
    # Equation (A3): parallel temperature
    dT_par_dt[:] = -6.0 * tau * model.Cperp(T_par)
    dT_par_dt -= 2/3 * tau * model.Cperp(P_parpar)
    dT_par_dt += 1.0 * tau * model.Cperp(P_parperp)
    model._temp1[:] = model.Cpar(q_par) - model.CparB(q_par)
    dT_par_dt -= 3.0 * sqrt_tau * model._temp1
    dT_par_dt += 2.0 * sqrt_tau * model.CparB(q_perp)
    dT_par_dt -= 2.0 * sqrt_tau * model.Cpar(u_par)
    dT_par_dt -= 0.5 * (1 - model.lperp) * model.iky * model.p.RT * phi
    
    # Equation (A4): perpendicular temperature
    dT_perp_dt[:] = -4.0 * tau * model.Cperp(T_perp)
    model._temp1[:] = N - 2 * P_parperp + 2 * P_perpperp
    dT_perp_dt += 1.0 * tau * model.Cperp(model._temp1)
    model._temp1[:] = model.Cpar(q_perp) - 2 * model.CparB(q_perp)
    dT_perp_dt -= sqrt_tau * model._temp1
    dT_perp_dt -= sqrt_tau * model.CparB(u_par)
    dT_perp_dt -= (model.lperp * model.iky * model.p.RN + (3 * model.lperp - 1) * model.iky * model.p.RT) * phi
    
    # Equation (A5): parallel heat flux
    model._temp1[:] = model.CparB(P_parpar) - model.Cpar(P_parpar)
    dq_par_dt[:] = 2.0 * sqrt_tau * model._temp1
    dq_par_dt += 3.0 * sqrt_tau * model.CparB(P_parperp)
    dq_par_dt -= 3.0 * sqrt_tau * model.Cpar(T_par)
    
    # Equation (A6): perpendicular heat flux
    model._temp1[:] = model.Cpar(P_parperp) - 2 * model.CparB(P_parperp)
    dq_perp_dt[:] = -2.0 * sqrt_tau * model._temp1
    dq_perp_dt += 2.0 * sqrt_tau * model.CparB(P_perpperp)
    model._temp1[:] = model.Cpar(T_perp) + model.CparB(T_perp)
    dq_perp_dt -= sqrt_tau * model._temp1
    dq_perp_dt -= 2.0 * sqrt_tau * model.CparB(T_par)
    
    # Compute the nonlinear terms using Poisson brackets if enabled
    if model.nonlinear:
        # Prepare modified potentials for Poisson brackets using pre-allocated arrays
        model._temp_phi1[:] = (1 - model.lperp) * phi
        model._temp_phi2[:] = model.lperp * phi

        # Equation (A1): density
        dN_dt -= model.pb.compute(model._temp_phi1, N)
        dN_dt -= model.pb.compute(model._temp_phi2, T_perp)
        
        # Equation (A2): parallel velocity
        du_par_dt -= model.pb.compute(model._temp_phi1, u_par)
        du_par_dt -= model.pb.compute(model._temp_phi2, q_perp)
        
        # Equation (A3): parallel temperature
        dT_par_dt -= model.pb.compute(model._temp_phi1, T_par)
        dT_par_dt -= model.pb.compute(model._temp_phi2, P_parperp)
        
        # Equation (A4): perpendicular temperature
        dT_perp_dt -= model.pb.compute(model._temp_phi1, T_perp)
        dT_perp_dt -= 0.5 * model.pb.compute(model._temp_phi2, P_perpperp)
        dT_perp_dt += tau * model.pb.compute(model._temp_phi1, N)
        
        # Equation (A5): parallel heat flux
        dq_par_dt -= model.pb.compute(phi, q_par)
        
        # Equation (A6): perpendicular heat flux
        dq_perp_dt -= model.pb.compute(phi, q_perp)
        dq_perp_dt += model.pb.compute(phi, u_par)
        
        # Equation (A7): parallel-parallel pressure tensor
        dP_parpar_dt -= model.pb.compute(phi, P_parpar)
        
        # Equation (A8): perpendicular-parallel pressure tensor
        dP_perppar_dt -= model.pb.compute(phi, P_parperp)
        dP_perppar_dt += model.pb.compute(phi, T_par)
        
        # Equation (A9): perpendicular-perpendicular pressure tensor
        dP_perpperp_dt -= model.pb.compute(phi, P_perpperp)
        dP_perpperp_dt += 0.5 * model.pb.compute(phi, T_perp)
        dP_perpperp_dt -= 0.25 * model.pb.compute(phi, N)
    
    # Return the derivatives, including dphidt=0
    return dydt
