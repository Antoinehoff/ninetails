# config.py
import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class PhysicalParams:
    # Core parameters
    tau: float  # Temperature ratio (T_i/T_e)
    RN: float   # Density gradient scale length
    RT: float   # Temperature gradient scale length
    
    # s-alpha geometry parameters
    eps: float = 0.1        # Inverse aspect ratio
    shear: float = 0.0      # Magnetic shear
    alpha_MHD: float = 0.0  # MHD alpha parameter
    q0: float = 2.0         # Safety factor
    R0: float = 1.0         # Major radius
    
    # Hasegawa-Wakatani parameters
    alpha: float = 0.5  # adiabaticity parameter
    kappa: float = 1.0  # curvature parameter

@dataclass
class NumericalParams:
    # Grid parameters
    nx: int       # Number of grid points in x
    ny: int       # Number of grid points in y
    nz: int       # Number of grid points in z
    Lx: float     # Domain size in x
    Ly: float     # Domain size in y
    Lz: float     # Domain size in z
    
    # Time stepping parameters
    dt: float     # Initial time step (for fixed-step methods)
    max_time: float  # Maximum simulation time
    n_outputs: int = 100  # Number of output points
    
    # Solver parameters
    atol: float = 1e-8  # Absolute tolerance for adaptive solver
    rtol: float = 1e-6  # Relative tolerance for adaptive solver
    muHD: float = 0.0  # Hyperdiffusion coefficient

@dataclass
class SimulationConfig:
    physical: PhysicalParams
    numerical: NumericalParams
    geometry_type: str  # 'salpha' or 'zpinch'
    nonlinear: bool
    output_dir: str
    input_file: str
    model_type: str # '9GM' or 'HW'
    nframes: int = 100
    restart: bool = False
    restart_file: Optional[str] = None
    save_restart: bool = False
    restart_interval: int = 10
    
    @classmethod
    def from_yaml(cls, filename):
        """
        Load configuration from a YAML file
        
        Parameters:
        -----------
        filename : str
            Path to the YAML configuration file
            
        Returns:
        --------
        SimulationConfig
            Configuration object
        """

        with open(filename, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Handle physical parameters with defaults
        phys_data = config_data.get('physical', {})
        phys = PhysicalParams(
            tau=phys_data.get('tau', 0.01),
            RN=phys_data.get('RN', 1.0),
            RT=phys_data.get('RT', 100.0),
            eps=phys_data.get('eps', 0.1),
            shear=phys_data.get('shear', 0.0),
            alpha_MHD=phys_data.get('alpha_MHD', 0.0),
            q0=phys_data.get('q0', 2.0),
            R0=phys_data.get('R0', 1.0),
            alpha=phys_data.get('alpha', 0.5),
            kappa=phys_data.get('kappa', 1.0)
        )
        
        # Handle numerical parameters
        num_data = config_data.get('numerical', {})
        num = NumericalParams(
            nx=num_data.get('nx', 64),
            ny=num_data.get('ny', 64),
            nz=num_data.get('nz', 16),
            Lx=num_data.get('Lx', 100.0),
            Ly=num_data.get('Ly', 100.0),
            Lz=num_data.get('Lz', 2.0),
            dt=num_data.get('dt', 0.01),
            max_time=num_data.get('max_time', 100.0),
            n_outputs=num_data.get('n_outputs', 100),
            atol=num_data.get('atol', 1e-8),
            rtol=num_data.get('rtol', 1e-6),
            muHD=num_data.get('muHD', 0.0)
        )
        
        # Create and return the config object
        return cls(
            physical=phys,
            numerical=num,
            geometry_type=config_data.get('geometry_type', 'zpinch'),
            nonlinear=config_data.get('nonlinear', True),
            model_type=config_data.get('model_type', '9GM'),
            input_file=filename,
            nframes=config_data.get('nframes', 100),
            output_dir=config_data.get('output_dir', 'output'),
            restart=config_data.get('restart', False),
            restart_file=config_data.get('restart_file', None),
            save_restart=config_data.get('save_restart', False),
            restart_interval=config_data.get('restart_interval', 10)
        )
    
    def save(self, filename):
        """
        Save configuration to a YAML file
        
        Parameters:
        -----------
        filename : str
            Path to save the YAML configuration file
        """
        # Convert dataclasses to dictionaries
        config_data = {
            'physical': asdict(self.physical),
            'numerical': asdict(self.numerical),
            'geometry_type': self.geometry_type,
            'nonlinear': self.nonlinear,
            'model_type': self.model_type,
            'nframes': self.nframes,
            'output_dir': self.output_dir,
            'restart': self.restart,
            'restart_file': self.restart_file,
            'save_restart': self.save_restart,
            'restart_interval': self.restart_interval
        }
        
        # Save to YAML file
        with open(filename, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        print(f"Configuration saved to {filename}")
    
    def create_default_config(filename):
        """
        Create a default configuration file
        
        Parameters:
        -----------
        filename : str
            Path to save the default configuration file
        """
        # Create default objects
        phys = PhysicalParams(
            tau=0.01,
            RN=1.0,
            RT=100.0
        )
        
        num = NumericalParams(
            nx=64,
            ny=64,
            nz=16,
            Lx=100.0,
            Ly=100.0,
            Lz=2.0,
            dt=0.01,
            max_time=100.0
        )
        
        config = SimulationConfig(
            physical=phys,
            numerical=num,
            geometry_type='zpinch',
            nonlinear=True,
            nframes=100,
            input_file=filename,
            output_dir='output',
            model_type= '9GM',
        )
        
        # Save to file
        config.save(filename)
