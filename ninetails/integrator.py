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
        
        # Pre-allocate work arrays for RK4 to avoid allocations in loop
        self.k1 = None
        self.k2 = None  
        self.k3 = None
        self.k4 = None
        self.y_temp = None
        self.rhs_temp = None
            
    def print_and_diag(self, t, y, dt):
        if self.verbose:
            print(f"t = {t:.2e}, dt = {dt:.2e}")
        self.diagnostic.update(t, y)
 
    def integrate(self, rhs_func, y0, t_span, dt, BC):
        """
        Integrate the ODE system using the specified method.
        """
        print(f"Integrating using {self.method} method...")
        
        start_time = time.time()  # Record start time
        
        if self.method in ['RK4', 'rk4']:
            t, y = self.rk4_integrate(rhs_func, t_span, y0, dt, BC)
            self.print_and_diag(t, y, dt)
        else:
            raise ValueError("Invalid integration method")
        
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        
        print(f"Integration completed in {elapsed_time:.2f} seconds.")
        
        self.diagnostic.finalize()
        
        return t, y

    def rk4_step(self, rhs, t, y, dt):
        """
        Optimized RK4 step with pre-allocated arrays to avoid allocations.
        """
        # Initialize work arrays on first call
        if self.k1 is None:
            self._allocate_work_arrays(y)
        
        # k1 = dt * rhs(t, y)
        rhs(t, y, self.rhs_temp)
        for i in range(len(y)):
            self.k1[i][:] = dt * self.rhs_temp[i]
        
        # k2 = dt * rhs(t + 0.5*dt, y + 0.5*k1)
        for i in range(len(y)):
            self.y_temp[i][:] = y[i] + 0.5 * self.k1[i]
        rhs(t + 0.5 * dt, self.y_temp, self.rhs_temp)
        for i in range(len(y)):
            self.k2[i][:] = dt * self.rhs_temp[i]
        
        # k3 = dt * rhs(t + 0.5*dt, y + 0.5*k2)
        for i in range(len(y)):
            self.y_temp[i][:] = y[i] + 0.5 * self.k2[i]
        rhs(t + 0.5 * dt, self.y_temp, self.rhs_temp)
        for i in range(len(y)):
            self.k3[i][:] = dt * self.rhs_temp[i]
        
        # k4 = dt * rhs(t + dt, y + k3)
        for i in range(len(y)):
            self.y_temp[i][:] = y[i] + self.k3[i]
        rhs(t + dt, self.y_temp, self.rhs_temp)
        for i in range(len(y)):
            self.k4[i][:] = dt * self.rhs_temp[i]
        
        # y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        for i in range(len(y)):
            y[i][:] += (self.k1[i] + 2*self.k2[i] + 2*self.k3[i] + self.k4[i]) / 6
    
    def _allocate_work_arrays(self, y):
        """Allocate work arrays for RK4 integration."""
        self.k1 = [np.zeros_like(yi) for yi in y]
        self.k2 = [np.zeros_like(yi) for yi in y]
        self.k3 = [np.zeros_like(yi) for yi in y]
        self.k4 = [np.zeros_like(yi) for yi in y]
        self.y_temp = [np.zeros_like(yi) for yi in y]
        self.rhs_temp = [np.zeros_like(yi) for yi in y]

    def rk4_integrate(self, rhs, t_span, y0, dt, BC):
        t = t_span[0]
        y = [yi.copy() for yi in y0]  # Only copy once at start
        tmax = t_span[1]
        while t < tmax:
            if t + dt > tmax:
                dt = tmax - t  # Adjust dt for the last step
            
            BC.apply(y)
            
            self.rk4_step(rhs, t, y, dt)  # y is modified in-place
            
            self.print_and_diag(t, y, dt)

            t += dt
            self.istep += 1
        return t, y