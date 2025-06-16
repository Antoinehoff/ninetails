#!/usr/bin/env python3
"""
Detailed debug script to analyze individual terms in GM3 equations
"""
import numpy as np
import sys
import os
sys.path.append('/Users/ahoffmann/programming/ninetails')

import ninetails as ntl

def analyze_gm3_terms():
    """
    Analyze individual terms in GM3 equations to find source of instability
    """
    print("=== GM3 Terms Analysis ===\n")
    
    # Setup the same configuration as before
    from ninetails.config import PhysicalParams, NumericalParams
    
    phys_params = PhysicalParams(
        RN=0.0, RT=0.0, tau=0.1, eps=0.1, shear=0.0, alpha_MHD=0.0, q0=2.0, R0=1.0, nu=0.1
    )
    num_params = NumericalParams(nx=32, ny=32, nz=1, Lx=20.0, Ly=20.0, max_time=100.0, muHD=0.001)
    config = ntl.SimulationConfig(phys_params, num_params)
    config.model_type = 'GM3'
    config.geometry_type = 'zpinch'
    config.nonlinear = False
    config.nframes = 100
    
    # Create simulation with zero magnetic gradients
    simulation = ntl.Simulation(input_file=None, config=config)
    simulation.setup()
    
    geometry = simulation.equations.geometry  
    geometry.dlnBdx[:, :, :] = 0.0
    geometry.dlnBdy[:, :, :] = 0.0
    geometry.dlnBdz[:, :, :] = 0.0
    geometry.get_curvature_operators()
    
    # Get a test state vector with small perturbations
    y_test = simulation.y0.copy()
    
    # Add small perturbation to density at the most unstable mode from before
    ikx_test = 8  # Corresponds to kx ≈ 0.157
    iky_test = 5  # Corresponds to ky ≈ 0.785  
    y_test[0][ikx_test, iky_test, 0] = 0.01 + 0.01j
    y_test[2][ikx_test, iky_test, 0] = 0.005 + 0.005j  # n20
    y_test[3][ikx_test, iky_test, 0] = 0.005 + 0.005j  # N01
    
    # Solve for potential
    y_test = simulation.equations.poisson_solver.solve(y_test)
    
    # Get the model and compute individual terms
    model = simulation.equations
    
    # Test GM3 function and analyze terms
    print(f"Analyzing terms at mode (kx={geometry.kx[ikx_test]:.3f}, ky={geometry.ky[iky_test]:.3f}):")
    
    # Extract moments
    N00, _, n20, N01, _, _, _, _, _, phi = y_test
    tau = model.p.tau
    sqrt2 = np.sqrt(2)
    n00 = N00 + model.K0 / tau * phi
    n01 = N01 + model.K1 / tau * phi
    
    print(f"\nMoment values at test mode:")
    print(f"  N00: {N00[ikx_test, iky_test, 0]}")
    print(f"  n20: {n20[ikx_test, iky_test, 0]}")
    print(f"  N01: {N01[ikx_test, iky_test, 0]}")
    print(f"  phi: {phi[ikx_test, iky_test, 0]}")
    print(f"  n00: {n00[ikx_test, iky_test, 0]}")
    print(f"  n01: {n01[ikx_test, iky_test, 0]}")
    
    # Analyze equation for N00 (density)
    print(f"\nEquation for N00 (density) terms:")
    
    # Term 1: Curvature term
    term1 = -tau * model.Cperp(2*n00 + sqrt2*n20 - n01)
    print(f"  Curvature term: {term1[ikx_test, iky_test, 0]:.6e}")
    
    # Term 2: Temperature gradient (should be zero)
    term2 = model.K1 * model.p.RT * model.iky * phi
    print(f"  Temperature grad term: {term2[ikx_test, iky_test, 0]:.6e}")
    
    # Term 3: Density gradient (should be zero)  
    term3 = -model.K0 * model.p.RN * model.iky * phi
    print(f"  Density grad term: {term3[ikx_test, iky_test, 0]:.6e}")
    
    # Term 4: Collision term
    term4 = -2./3. * model.p.nu * model.lperp * (5*n00 + sqrt2 * n20)
    print(f"  Collision term: {term4[ikx_test, iky_test, 0]:.6e}")
    
    # Term 5: Hyperdiffusion
    term5 = -model.p.muHD * model.kperp2 * y_test[0]
    print(f"  Hyperdiffusion term: {term5[ikx_test, iky_test, 0]:.6e}")
    
    # Check if collision term has wrong sign or magnitude
    print(f"\nCollision term details:")
    print(f"  nu: {model.p.nu}")
    print(f"  lperp at test mode: {model.lperp[ikx_test, iky_test, 0]:.6f}")
    print(f"  (5*n00 + sqrt2*n20): {(5*n00 + sqrt2*n20)[ikx_test, iky_test, 0]:.6e}")
    
    # Analyze equation for n20 (parallel temperature)
    print(f"\nEquation for n20 (parallel temperature) terms:")
    
    term1_n20 = tau * model.Cperp(6*n20 + sqrt2*n00)
    term2_n20 = -0.5 * sqrt2 * model.p.RT * model.iky * model.K0 * phi
    term3_n20 = -2./3. * model.p.nu * (2*sqrt2*n00 + 2*n20 + sqrt2*n01)
    term4_n20 = -model.p.muHD * model.kperp2 * y_test[2]
    
    print(f"  Curvature term: {term1_n20[ikx_test, iky_test, 0]:.6e}")
    print(f"  Temperature grad term: {term2_n20[ikx_test, iky_test, 0]:.6e}")
    print(f"  Collision term: {term3_n20[ikx_test, iky_test, 0]:.6e}")
    print(f"  Hyperdiffusion term: {term4_n20[ikx_test, iky_test, 0]:.6e}")
    
    # Analyze equation for N01 (perpendicular temperature)
    print(f"\nEquation for N01 (perpendicular temperature) terms:")
    
    term1_N01 = tau * model.Cperp(4*n01 - n00)
    term2_N01 = -model.p.RT * model.iky * (2*model.K1 - model.K0 - 2*model.K2) * phi
    term3_N01 = -model.p.RN * model.iky * model.K1 * phi
    term4_N01 = -2./3. * model.p.nu * (2*n00 + sqrt2*n20 + n01)
    term5_N01 = -model.p.muHD * model.kperp2 * y_test[3]
    
    print(f"  Curvature term: {term1_N01[ikx_test, iky_test, 0]:.6e}")
    print(f"  Temperature grad term: {term2_N01[ikx_test, iky_test, 0]:.6e}")
    print(f"  Density grad term: {term3_N01[ikx_test, iky_test, 0]:.6e}")
    print(f"  Collision term: {term4_N01[ikx_test, iky_test, 0]:.6e}")
    print(f"  Hyperdiffusion term: {term5_N01[ikx_test, iky_test, 0]:.6e}")

if __name__ == "__main__":
    analyze_gm3_terms()
