# rk45.py

import numpy as np
import time  # Add this import

class Integrator:
    """
    Class for integrating ODE systems using various methods.
    
    Parameters:
    -----------
    method : str
        Integration method to use. Options are 'RK4' and 'RK45'.
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance
    verbose : bool
        Print diagnostic information
    diagnostic : object
        Diagnostic object for monitoring the solution
    """
    method = None
    atol = None
    rtol = None   
    verbose = None
    diagnostic = None
    nprint = 0
    istep = 0
    
    def __init__(self, method='RK4', atol=1e-8, rtol=1e-6, verbose=False, diagnostic=None):
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.verbose = verbose
        self.diagnostic = diagnostic
            
    def print_and_diag(self, t, y, dt):
        if self.verbose:
            print(f"t = {t:.2e}, dt = {dt:.2e}")
        self.diagnostic.update(t, y)
 
    def integrate(self, rhs_func, y0, t_span, dt):
        """
        Integrate the ODE system using the specified method.
        """
        print(f"Integrating using {self.method} method...")
        
        start_time = time.time()  # Record start time
        
        if self.method in ['RK4', 'rk4']:
            t, y = self.rk4_integrate(rhs_func, t_span, y0, dt)
            self.print_and_diag(t, y, dt)
        else:
            raise ValueError("Invalid integration method")
        
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        
        print(f"Integration completed in {elapsed_time:.2f} seconds.")
        
        self.diagnostic.finalize()
        
        return t, y

    def rk4_step(self, rhs, t, y, dt):
        k1 = dt * rhs(t, y)
        k2 = dt * rhs(t + 0.5 * dt, y + 0.5 * k1)
        k3 = dt * rhs(t + 0.5 * dt, y + 0.5 * k2)
        k4 = dt * rhs(t + dt, y + k3)
        
        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        return y_next

    def rk4_integrate(self, rhs, t_span, y0, dt):
        t = t_span[0]
        y = y0.copy()
        tmax = t_span[1]
        while t < tmax:
            if t + dt > tmax:
                dt = tmax - t  # Adjust dt for the last step
            
            y = self.rk4_step(rhs, t, y, dt)
            
            self.print_and_diag(t, y, dt)

            t += dt
            self.istep += 1
        return t, y