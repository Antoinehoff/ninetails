import h5py
from .config import SimulationConfig
from .diagnostics import Diagnostics

def save_solution_hdf5(filename, diagnostics, config):
    """
    Save the simulation solution and configuration to an HDF5 file.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file where the solution will be saved
    solution : Bunch object
        Solution object returned by solve_ivp
    diagnostics : Diagnostics
        Diagnostics object containing energy and enstrophy history
    config : SimulationConfig
        Configuration object used for the simulation
    """
    with h5py.File(filename, 'w') as f:
        
        # Save diagnostics data
        diag_group = f.create_group('diagnostics')
        
        # Save frames
        t = diagnostics.frames['t']
        fields = diagnostics.frames['fields']
        diag_group.create_dataset('frames/t', data=t)
        diag_group.create_dataset('frames/fields', data=fields)
        
        # Save energy history
        t = diagnostics.energy_history['t']
        Etot = diagnostics.energy_history['total']
        Ekin = diagnostics.energy_history['kinetic']
        Etherm = diagnostics.energy_history['thermal']
        Epot = diagnostics.energy_history['potential']
        diag_group.create_dataset('energy_history/t', data=t)
        diag_group.create_dataset('energy_history/total', data=Etot)
        diag_group.create_dataset('energy_history/kinetic', data=Ekin)
        diag_group.create_dataset('energy_history/thermal', data=Etherm)
        diag_group.create_dataset('energy_history/potential', data=Epot)
        
        # save enstrophy history
        t = diagnostics.enstrophy_history['t']
        enstrophy = diagnostics.enstrophy_history['enstrophy']
        diag_group.create_dataset('enstrophy_history/t', data=t)
        diag_group.create_dataset('enstrophy_history/enstrophy', data=enstrophy)
        
        # Add the input file .yaml to the HDF5 file as text file
        with open(config.input_file, 'r') as input_file:
            f.create_dataset('input_file', data=input_file.read())
        
    print(f"Solution and configuration saved to {filename}")

def load_solution_hdf5(filename):
    """
    Load the simulation solution and configuration from an HDF5 file.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file where the solution is saved
        
    Returns:
    --------
    tuple
        (solution, diagnostics, config) where solution is a dictionary containing the time and state vectors,
        diagnostics is a dictionary containing energy and enstrophy history,
        and config is the SimulationConfig object
    """
    with h5py.File(filename, 'r') as f:        
        # Load diagnostics data
        diagnostics = Diagnostics()
        
        diagnostics.frames = {
            't': f['diagnostics/frames/t'][:],
            'fields': f['diagnostics/frames/fields'][:]
        }
        
        diagnostics.energy_history = {
            't': f['diagnostics/energy_history/t'][:],
            'total': f['diagnostics/energy_history/total'][:],
            'kinetic': f['diagnostics/energy_history/kinetic'][:],
            'thermal': f['diagnostics/energy_history/thermal'][:],
            'potential': f['diagnostics/energy_history/potential'][:]
        }
        diagnostics.enstrophy_history = {
            't': f['diagnostics/enstrophy_history/t'][:],
            'enstrophy': f['diagnostics/enstrophy_history/enstrophy'][:]
        }
        
        # Load configuration from the input file stored in the HDF5 file
        input_file_content = f['input_file'][()]
        config_filename = filename.replace('.h5', '_config.yaml')
        
        with open(config_filename, 'w') as config_file:
            config_file.write(input_file_content.decode('utf-8'))
        
        config = SimulationConfig.from_yaml(config_filename)
    
    print(f"Solution, diagnostics, and configuration loaded from {filename}")
    return diagnostics, config