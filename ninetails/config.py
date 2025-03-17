# config.py
import yaml
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class PhysicalParams:
    # Core parameters
    tau: float = 0.001  # Temperature ratio (T_i/T_e)
    RN: float = 1.0  # Density gradient scale length
    RT: float = 3500  # Temperature gradient scale length
    
    # s-alpha geometry parameters
    eps: float = 0.1        # Inverse aspect ratio
    shear: float = 0.8      # Magnetic shear
    alpha_MHD: float = 0.0  # MHD alpha parameter
    q0: float = 1.4         # Safety factor
    R0: float = 1.0         # Major radius
    
    # Hasegawa-Wakatani parameters
    alpha: float = 0.1  # adiabaticity parameter
    kappa: float = 1.0  # curvature parameter

    def info(self):
        print("Physical Parameters:")
        for key, value in asdict(self).items():
            print(f"  {key}: {value}")

@dataclass
class NumericalParams:
    # Grid parameters
    nx: int = 64            # Number of grid points in x
    ny: int = 64            # Number of grid points in y
    nz: int = 1             # Number of grid points in z
    Lx: float = 40.0        # Domain size in x
    Ly: float = 40.0        # Domain size in y
    Lz: float = 2.0*np.pi   # Domain size in z
    
    # Time stepping parameters
    dt: float = 0.1         # Initial time step (for fixed-step methods)
    max_time: float = 50.0  # Maximum simulation time
    
    # Solver parameters
    atol: float = 1e-8  # Absolute tolerance for adaptive solver
    rtol: float = 1e-6  # Relative tolerance for adaptive solver
    muHD: float = 0.01  # Hyperdiffusion coefficient

    def info(self):
        print("Numerical Parameters:")
        for key, value in asdict(self).items():
            print(f"  {key}: {value}")

@dataclass
class SimulationConfig:
    physical: PhysicalParams
    numerical: NumericalParams
    geometry_type: str = 'zpinch'   # 'salpha' or 'zpinch'
    nonlinear: bool = True
    output_dir: str = 'output'
    input_file: str = 'input.yaml'
    model_type: str = 'MHW'         # '9GM', 'HW', or 'MHW'
    nframes: int = 50
    restart: bool = False
    restart_frame: int = 0
    restart_file: str = None
    follow_frame: bool = False
    
    @classmethod
    def from_yaml(cls, filename):
        with open(filename, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Handle physical parameters with defaults
        phys_data = config_data.get('physical', {})
        phys = PhysicalParams(
            tau=phys_data.get('tau', 0.001),
            RN=phys_data.get('RN', 1.0),
            RT=phys_data.get('RT', 3500.0),
            eps=phys_data.get('eps', 0.1),
            shear=phys_data.get('shear', 0.0),
            alpha_MHD=phys_data.get('alpha_MHD', 0.0),
            q0=phys_data.get('q0', 1.4),
            R0=phys_data.get('R0', 1.0),
            alpha=phys_data.get('alpha', 0.1),
            kappa=phys_data.get('kappa', 1.0)
        )
        
        # Handle numerical parameters
        num_data = config_data.get('numerical', {})
        num = NumericalParams(
            nx=num_data.get('nx', 64),
            ny=num_data.get('ny', 64),
            nz=num_data.get('nz', 1),
            Lx=num_data.get('Lx', 40),
            Ly=num_data.get('Ly', 40.0),
            Lz=num_data.get('Lz', 2.0),
            dt=num_data.get('dt', 0.1),
            max_time=num_data.get('max_time', 100.0),
            muHD=num_data.get('muHD', 0.01)
        )
        
        # Create and return the config object
        return cls(
            physical=phys,
            numerical=num,
            geometry_type=config_data.get('geometry_type', 'zpinch'),
            nonlinear=config_data.get('nonlinear', True),
            model_type=config_data.get('model_type', 'MHW'),
            input_file=filename,
            output_dir=config_data.get('output_dir', 'output'),
            nframes=config_data.get('nframes', 100),
            restart=config_data.get('restart', False),
            restart_frame=config_data.get('restart_frame', 0),
            restart_file=config_data.get('restart_file', None),
            follow_frame=config_data.get('follow_frame', False)
        )
    
    def save(self, filename):
        # Convert dataclasses to dictionaries
        config_data = {
            'physical': asdict(self.physical),
            'numerical': asdict(self.numerical),
            'geometry_type': self.geometry_type,
            'nonlinear': self.nonlinear,
            'model_type': self.model_type,
            'output_dir': self.output_dir,
            'nframes': self.nframes,
            'restart': self.restart,
            'restart_frame': self.restart_frame,
            'restart_file': self.restart_file,
            'follow_frame': self.follow_frame
        }
        
        # Save to YAML file
        with open(filename, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False,
                      sort_keys=False)
            
    def create_default_config(filename):
        # Create default objects
        phys = PhysicalParams()
        
        num = NumericalParams()
        
        config = SimulationConfig(physical=phys, numerical=num)
        
        # Save to file
        config.save(filename)

        return config

    def info(self):
        print("Simulation Configuration:")
        print(f"  Geometry Type: {self.geometry_type}")
        print(f"  Nonlinear: {self.nonlinear}")
        print(f"  Model Type: {self.model_type}")
        print(f"  Input File: {self.input_file}")
        print(f"  Output Directory: {self.output_dir}")
        print(f"  Number of Frames: {self.nframes}")
        print(f"  Restart: {self.restart}")
        print(f"  Restart File: {self.restart_file}")
        self.physical.info()
        self.numerical.info()
