#!/usr/bin/env python3
"""
Debug script for GM3 solver to identify sources of instability
"""
import numpy as np
import sys
import os
sys.path.append('/Users/ahoffmann/programming/ninetails')

import ninetails as ntl

def debug_gm3_solver():
    """
    Test GM3 solver with zero gradients to check for spurious instabilities
    """
    print("=== GM3 Debug Analysis ===\n")
    
    # Create a test configuration with zero gradients
    from ninetails.config import PhysicalParams, NumericalParams
    
    # Set zero gradients
    phys_params = PhysicalParams(
        RN=0.0,     # Zero density gradient
        RT=0.0,     # Zero temperature gradient  
        tau=0.1,    # Reasonable tau value
        eps=0.1,
        shear=0.0,
        alpha_MHD=0.0,
        q0=2.0,
        R0=1.0,
        nu=0.1
    )
    
    num_params = NumericalParams(
        nx=32,
        ny=32,
        nz=1,
        Lx=20.0,
        Ly=20.0,
        max_time=100.0,
        muHD=0.001  # Small hyperdiffusion
    )
    
    config = ntl.SimulationConfig(phys_params, num_params)
    config.model_type = 'GM3'
    config.geometry_type = 'zpinch'
    config.nonlinear = False  # Linear analysis first
    config.nframes = 100
    
    print(f"Parameters:")
    print(f"  RN (density gradient): {config.physical.RN}")
    print(f"  RT (temperature gradient): {config.physical.RT}")
    print(f"  tau: {config.physical.tau}")
    print(f"  Model: {config.model_type}")
    print(f"  Geometry: {config.geometry_type}")
    print(f"  Nonlinear: {config.nonlinear}")
    
    # Create simulation
    simulation = ntl.Simulation(input_file=None, config=config)
    simulation.setup()
    
    # Override magnetic gradients to zero for true stability test
    geometry = simulation.equations.geometry
    geometry.dlnBdx[:, :, :] = 0.0
    geometry.dlnBdy[:, :, :] = 0.0  
    geometry.dlnBdz[:, :, :] = 0.0
    # Recompute curvature operators with zero magnetic gradients
    geometry.get_curvature_operators()
    
    print(f"\nModified geometry for stability test:")
    print(f"  dlnBdx set to: 0.0 (was -1.0)")
    print(f"  dlnBdy set to: 0.0")
    print(f"  dlnBdz set to: 0.0")
    
    # Check the geometry and operators
    geometry = simulation.equations.geometry
    print(f"\nGeometry check:")
    print(f"  kx range: [{np.min(geometry.kx):.3f}, {np.max(geometry.kx):.3f}]")
    print(f"  ky range: [{np.min(geometry.ky):.3f}, {np.max(geometry.ky):.3f}]")
    print(f"  K0 range: [{np.min(geometry.K0.real):.3f}, {np.max(geometry.K0.real):.3f}]")
    print(f"  K1 range: [{np.min(geometry.K1.real):.3f}, {np.max(geometry.K1.real):.3f}]")
    print(f"  lperp range: [{np.min(geometry.lperp):.3f}, {np.max(geometry.lperp):.3f}]")
    
    # Test the curvature operators on a sample field
    sample_field = np.ones_like(geometry.kperp2, dtype=np.complex128)
    cperp_result = geometry.Cperp(sample_field)
    print(f"  Cperp operator test: max|result| = {np.max(np.abs(cperp_result)):.3e}")
    
    # Check if Cperp is causing issues by examining its structure
    print(f"  Cxy operator: max|value| = {np.max(np.abs(geometry.Cxy)):.3e}")
    
    # Run a short simulation
    print(f"\nRunning short simulation...")
    
    try:
        simulation.run()
        print("✓ Simulation completed successfully")
        
        # Analyze growth rates
        from ninetails.plotter import Plotter
        plotter = Plotter(simulation)
        print("\nAnalyzing growth rates...")
        
        # Try to compute growth rates for density
        try:
            plotter.growth_rates(moment_name='N00', filename='debug_growth_rates.png')
            print("✓ Growth rate analysis completed")
            
            # Get the actual growth rate values for inspection
            time, field = simulation.diagnostics.get_moment_data('N00')
            growth_rates, _, errors = ntl.PostProcessor.compute_growth_rates(
                simulation, time, field, return_error=True
            )
            
            print(f"\nGrowth rate statistics:")
            print(f"  Max growth rate: {np.max(growth_rates):.6f}")
            print(f"  Min growth rate: {np.min(growth_rates):.6f}")
            print(f"  Mean growth rate: {np.mean(growth_rates):.6f}")
            print(f"  Std growth rate: {np.std(growth_rates):.6f}")
            
            # Check if any modes are unstable
            unstable_modes = np.sum(growth_rates > 1e-6)
            print(f"  Number of unstable modes (γ > 1e-6): {unstable_modes}")
            
            if unstable_modes > 0:
                print("⚠️  WARNING: Found unstable modes with zero gradients!")
                # Find the most unstable mode
                ikx_max, iky_max = np.unravel_index(np.argmax(growth_rates), growth_rates.shape)
                kx_unstable = geometry.kx[ikx_max]
                ky_unstable = geometry.ky[iky_max]
                print(f"     Most unstable mode: kx={kx_unstable:.3f}, ky={ky_unstable:.3f}")
                print(f"     Growth rate: {growth_rates[ikx_max, iky_max]:.6f}")
            else:
                print("✓ No unstable modes found (as expected with zero gradients)")
                
        except Exception as e:
            print(f"✗ Growth rate analysis failed: {e}")
            
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = debug_gm3_solver()
    if success:
        print("\n=== Debug completed ===")
    else:
        print("\n=== Debug failed ===")
        sys.exit(1)
