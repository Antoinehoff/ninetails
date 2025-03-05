# rk45.py

import numpy as np

def rk45_step(rhs_func, t, y, dt):
    """
    Perform a single step of the RK45 integration method.
    
    Parameters:
    -----------
    rhs_func : callable
        Function that computes the right-hand side of the ODE.
        Should take arguments (t, y) and return dy/dt.
    t : float
        Current time.
    y : ndarray
        Current state vector.
    dt : float
        Time step size.
        
    Returns:
    --------
    y_next : ndarray
        State vector after one RK45 step.
    """
    k1 = dt * rhs_func(t, y)
    k2 = dt * rhs_func(t + 1/4 * dt, y + 1/4 * k1)
    k3 = dt * rhs_func(t + 3/8 * dt, y + 3/32 * k1 + 9/32 * k2)
    k4 = dt * rhs_func(t + 12/13 * dt, y + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
    k5 = dt * rhs_func(t + dt, y + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
    k6 = dt * rhs_func(t + 1/2 * dt, y - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
    
    y_next = y + 25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5
    
    return y_next

def rk45_integrate(rhs_func, t_span, y0, t_eval, atol=1e-8, rtol=1e-6):
    """
    Integrate the ODE system using RK45 method.
    
    Parameters:
    -----------
    rhs_func : callable
        Function that computes the right-hand side of the ODE.
        Should take arguments (t, y) and return dy/dt.
    t_span : tuple
        Time span (t0, tf).
    y0 : ndarray
        Initial state vector.
    t_eval : ndarray
        Array of time points at which to output the solution.
    atol : float, optional
        Absolute tolerance for adaptive time stepping (default: 1e-8).
    rtol : float, optional
        Relative tolerance for adaptive time stepping (default: 1e-6).
        
    Returns:
    --------
    solution : dict
        Dictionary containing 't' (time points) and 'y' (state vectors at each time point).
    """
    t0, tf = t_span
    t = t0
    y = y0.copy()
    dt = t_eval[1] - t_eval[0]  # Assuming evenly spaced t_eval
    
    istep = 0
    while t < tf:
        if t + dt > tf:
            dt = tf - t  # Adjust dt for the last step
        
        # # display t and dt in scientific notation
        # if istep % 100 == 0:
        #     print(f"t = {t:.2e}, dt = {dt:.2e}")
        
        y_next = rk45_step(rhs_func, t, y, dt)
        
        # Error estimation
        y_next_hat = rk45_step(rhs_func, t, y, dt/2)
        error = np.max(np.abs(y_next - y_next_hat))
        
        # Adapt step size based on error
        if error < atol + rtol * np.max(np.abs(y_next)):
            y = y_next
            t += dt
        
        # Adjust dt based on error and tolerance
        if error == 0:
            s = 2.0
        else:
            s = 0.84 * (atol / error) ** 0.25
        
        dt *= s
        
        istep += 1
        
    return

def rk4_step(rhs_func, t, y, dt):
    """
    Perform a single step of the RK4 integration method.
    
    Parameters:
    -----------
    rhs_func : callable
        Function that computes the right-hand side of the ODE.
        Should take arguments (t, y) and return dy/dt.
    t : float
        Current time.
    y : ndarray
        Current state vector.
    dt : float
        Time step size.
        
    Returns:
    --------
    y_next : ndarray
        State vector after one RK4 step.
    """
    k1 = dt * rhs_func(t, y)
    k2 = dt * rhs_func(t + 0.5 * dt, y + 0.5 * k1)
    k3 = dt * rhs_func(t + 0.5 * dt, y + 0.5 * k2)
    k4 = dt * rhs_func(t + dt, y + k3)
    
    y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return y_next

def rk4_integrate(rhs_func, t_span, dt, y0, t_eval):
    """
    Integrate the ODE system using RK4 method.
    
    Parameters:
    -----------
    rhs_func : callable
        Function that computes the right-hand side of the ODE.
        Should take arguments (t, y) and return dy/dt.
    t_span : tuple
        Time span (t0, tf).
    y0 : ndarray
        Initial state vector.
    t_eval : ndarray
        Array of time points at which to output the solution.
        
    Returns:
    --------
    solution : dict
        Dictionary containing 't' (time points) and 'y' (state vectors at each time point).
    """
    t0, tf = t_span
    t = t0
    y = y0.copy()
    
    solution_t = [t]
    solution_y = [y.copy()]
    
    while t < tf:
        if t + dt > tf:
            dt = tf - t  # Adjust dt for the last step
        
        y = rk4_step(rhs_func, t, y, dt)
        t += dt
        
        solution_t.append(t)
        solution_y.append(y.copy())
    
    solution = {
        't': np.array(solution_t),
        'y': np.array(solution_y)
    }
    
    return solution

def integrate(method, rhs_func, dt, y0, t_eval, atol=1e-8, rtol=1e-6):
    if method =='RK4':
        return rk4_integrate(rhs_func, t_eval, dt, y0, t_eval)
    elif method == 'RK45':
        return rk45_integrate(rhs_func, t_eval, y0, t_eval, atol, rtol)
    else:
        raise ValueError("Invalid integration method")