# Ninetails: A Plasma Fluid Simulation Framework

Ninetails is a Python-based framework for plasma physics simulations using a high-order fluid model derived from the gyro-moment approach. The code is designed to simulate plasma instabilities and turbulence in both linear and nonlinear regimes, with support for multiple geometry configurations.

## Overview

This framework solves a set of fluid equations derived in the gyro-moment approach, consisting of 9 coupled partial differential equations for plasma moments (hence the name "Ninetails"):

1. Density (n)
2. Parallel velocity (u_parallel)
3. Parallel temperature (T_parallel)
4. Perpendicular temperature (T_perp)
5. Parallel heat flux (q_parallel)
6. Perpendicular heat flux (q_perp)
7. Parallel-parallel pressure tensor (P_parallel^parallel)
8. Parallel-perpendicular pressure tensor (P_parallel^perp)
9. Perpendicular-perpendicular pressure tensor (P_perp^perp)

These equations are complemented by a quasineutrality condition for the electrostatic potential (phi).

## Features

- **Multiple Geometry Types**: Support for s-alpha (tokamak) and Z-pinch geometries
- **Linear and Nonlinear Simulations**: Toggle nonlinear effects on/off
- **Pseudospectral Method**: Fast spectral methods for spatial discretization with anti-aliasing
- **Adaptive Time-Stepping**: Using SciPy's ODE solvers
- **Diagnostics**: Energy and enstrophy tracking, growth rate analysis
- **Post-Processing**: Visualization tools for simulation outputs
- **Configuration System**: YAML-based configuration for easy parameter adjustments

## Directory Structure

```
ninetails/
├── src/                   # Source code directory
│   ├── main.py            # Main simulation entry point
│   ├── config.py          # Configuration handling
│   ├── models.py          # Implementation of the fluid equations
│   ├── geometry.py        # Geometry specifications
│   ├── poisson_solver.py  # Solver for quasineutrality equation
│   ├── poisson_bracket.py # Poisson bracket calculator
│   ├── diagnostics.py     # Energy and enstrophy calculation
│   └── post_processing.py # Visualization and analysis tools
├── doc/                   # Documentation directory
│   └── equations.tex      # Mathematical formulation of the model
├── simulation_config.yaml # Default simulation parameters
├── README.md              # This file
└── LICENSE                # License information
```

## Installation

This project requires Python 3.7+ and several scientific computing packages:

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scipy matplotlib pyyaml
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/username/ninetails.git
cd ninetails
```

2. Run a simulation with default parameters:
```bash
python src/main.py
```

3. Check the output directory for results and visualizations:
```bash
ls output/figures/
```

## Configuration

The simulation parameters can be customized by editing the `simulation_config.yaml` file. Here's an example:

```yaml
# Physical parameters
physical:
  tau: 1.0              # Temperature ratio (T_e/T_i)
  RN: 2.0               # Density gradient scale length
  RT: 7.0               # Temperature gradient scale length
  eps: 0.1              # Inverse aspect ratio (for s-alpha geometry)
  shear: 0.0            # Magnetic shear (for s-alpha geometry)
  alpha_MHD: 0.0        # MHD alpha parameter (for s-alpha geometry)
  q0: 2.0               # Safety factor (for s-alpha geometry)
  R0: 1.0               # Major radius (for s-alpha geometry)
  
# Numerical parameters
numerical:
  nx: 33                # Number of grid points in x
  ny: 32                # Number of grid points in y
  nz: 1                 # Number of grid points in z
  Lx: 100.0             # Domain size in x
  Ly: 100.0             # Domain size in y
  Lz: 2.0               # Domain size in z
  dt: 0.001             # Initial time step
  max_time: 10.0        # Maximum simulation time
  hyperdiffusion: 0.1   # Hyperdiffusion coefficient
  
geometry_type: 'zpinch' # 'salpha' or 'zpinch'
nonlinear: false        # Include nonlinear terms
output_dir: 'output'    # Directory for output files
```

## Mathematical Model

The fluid model equations are implemented based on the formulation in `doc/equations.tex`. The main equations are:

1. **Density Evolution**: 
   ```math
   \partial_t n + \{(1 - \ell_\perp)\phi, n\} + \{\ell_\perp \phi, T_\perp\} + 2\tau \mathcal{C}_\perp(T_\parallel - T_\perp + n) + (\mathcal{C}_\parallel - \mathcal{C}_\parallel^B)\sqrt{\tau}\, u_\parallel + \left[(1 - \ell_\perp)i k_y R_N - \ell_\perp i k_y R_T\right]\phi = 0
   ```

2. **Parallel Velocity Evolution**:
   ```math
   \partial_t u_\parallel + \{(1 - \ell_\perp)\phi, u_\parallel\} + \{\ell_\perp \phi, q_\perp\} + n \mathcal{C}_\parallel\sqrt{\tau} + 4 \tau \mathcal{C}_\perp u_\parallel + 6\tau \mathcal{C}_\perp q_\parallel - \tau \mathcal{C}_\perp q_\perp + 2(\mathcal{C}_\parallel - \mathcal{C}_\parallel^B)\sqrt{\tau} T_\parallel - \mathcal{C}_\parallel^B\sqrt{\tau} T_\perp = 0
   ```

(and so on for the remaining equations)

The equations are complemented by the quasineutrality condition:
$$\left(1 - 2\left[\ell_\perp - \tau\ell_\perp^2\right]\right)\phi - \langle\phi\rangle_{yz} = n + \tau\ell_\perp(T_\perp - n)$$

## Running Advanced Simulations

### Linear Stability Analysis

For linear stability analysis, set `nonlinear: false` in the configuration file. This will track the growth of individual modes and compute growth rates.

### Nonlinear Turbulence

For turbulence simulations, set `nonlinear: true`. Increase the resolution (`nx`, `ny`) and adjust physical parameters like gradient scale lengths (`RN`, `RT`).

### Parameter Scans

To perform parameter scans, modify the `src/main.py` file to loop over different parameter values. Example:

```python
# In src/main.py
RN_values = [1.0, 2.0, 5.0, 10.0]
for RN in RN_values:
    config.physical.RN = RN
    run_simulation(config)
```

## Visualization

The `src/post_processing.py` module provides several visualization functions:

- `plot_2D_snapshot`: 2D contour plots of fields
- `plot_time_evolution`: Time evolution of specific modes
- `plot_energy_evolution`: Energy components over time
- `plot_enstrophy_evolution`: Enstrophy evolution
- `compute_growth_rates`: Linear growth rate analysis

## Contributing

Contributions to Ninetails are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This code implements the fluid model developed in [reference to relevant papers].
