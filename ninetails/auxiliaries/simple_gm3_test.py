#!/usr/bin/env python3
"""
Simple test to isolate the GM3 instability source
"""
import numpy as np
import sys
import os
sys.path.append('/Users/ahoffmann/programming/ninetails')

import ninetails as ntl

def simple_gm3_test():
    """
    Test GM3 with minimal setup to isolate instability
    """
    print("=== Simple GM3 Test ===\n")
    
    # Create minimal config
    from ninetails.config import PhysicalParams, NumericalParams
    
    # Test with no collisions first
    phys_params = PhysicalParams(
        RN=0.0, RT=0.0, tau=0.1, eps=0.1, shear=0.0, alpha_MHD=0.0, 
        q0=2.0, R0=1.0, nu=0.0  # NO COLLISIONS
    )
    num_params = NumericalParams(
        nx=16, ny=16, nz=1, Lx=10.0, Ly=10.0, max_time=50.0, muHD=0.01  # Higher diffusion
    )
    
    config = ntl.SimulationConfig(phys_params, num_params)
    config.model_type = 'GM3'
    config.geometry_type = 'zpinch'
    config.nonlinear = False
    config.nframes = 50
    
    print("Testing GM3 with:")
    print(f"  RN = {phys_params.RN} (density grad)")
    print(f"  RT = {phys_params.RT} (temp grad)")
    print(f"  nu = {phys_params.nu} (collision freq)")
    print(f"  muHD = {num_params.muHD} (hyperdiffusion)")
    
    # Create simulation
    simulation = ntl.Simulation(input_file=None, config=config)
    simulation.setup()
    
    # Zero all magnetic gradients
    geometry = simulation.equations.geometry
    geometry.dlnBdx[:, :, :] = 0.0
    geometry.dlnBdy[:, :, :] = 0.0
    geometry.dlnBdz[:, :, :] = 0.0
    geometry.get_curvature_operators()
    
    print(f"  All magnetic gradients set to zero")
    print(f"  Cperp max: {np.max(np.abs(geometry.Cxy)):.6e}")
    
    # Run simulation
    simulation.run()
    
    # Check growth rates
    from ninetails.plotter import Plotter
    plotter = Plotter(simulation)
    
    time, field = simulation.diagnostics.get_moment_data('N00')
    growth_rates, _, _ = ntl.PostProcessor.compute_growth_rates(
        simulation, time, field, return_error=True
    )
    
    max_growth = np.max(growth_rates)
    unstable_modes = np.sum(growth_rates > 1e-6)
    
    print(f"\nResults:")
    print(f"  Max growth rate: {max_growth:.6f}")
    print(f"  Unstable modes: {unstable_modes}")
    
    if unstable_modes == 0:
        print("✓ STABLE - No spurious instabilities!")
        return True
    else:
        print("✗ UNSTABLE - Still has spurious instabilities")
        return False

if __name__ == "__main__":
    stable = simple_gm3_test()
    if not stable:
        print("\nThe instability is NOT from collisions or magnetic gradients.")
        print("It's likely from:")
        print("1. Error in GM3 equation implementation")
        print("2. Incorrect coupling between moments")
        print("3. Sign error in physics terms")
        print("4. Issue with Poisson solver or boundary conditions")
