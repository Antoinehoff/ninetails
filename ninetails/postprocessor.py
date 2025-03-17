# post_processing.py
import numpy as np

class PostProcessor:

    def to_real_space(simulation, data_hat):
        # Check if we have a time dimension
        has_time = len(data_hat.shape) > 3
        nx = simulation.ndims[0]
        ny = simulation.ndims[1]
        nz = simulation.ndims[2]

        if has_time:
            nt = data_hat.shape[0]
            data_real = np.zeros((nt, nx, ny, nz))
            
            # Loop over time and z
            for it in range(nt):
                for iz in range(nz):
                    data_real[it, :, :, iz] = np.squeeze(np.fft.irfft2(data_hat[it, :, :, iz], s=(nx, ny)))
        else:
            data_real = np.zeros((nx, ny, nz))
            
            # Loop over z
            for iz in range(nz):
                data_real[:, :, iz] = np.fft.irfft2(data_hat[:, :, iz], s=(nx, ny))
        
        return data_real

    def compute_growth_rates(simulation, time, field, tlim =[], z_idx=0):
        # Extract time and field data
        if tlim:
            time_indices = np.where((time >= tlim[0]) & (time <= tlim[1]))[0]
        else:
            # take half of the time
            time_indices = np.arange(len(time)//2)
        time = time[time_indices]
        field = field[time_indices, :, :, :]

        # Initialize growth rate array
        growth_rates = np.zeros((simulation.nkdims[0], simulation.nkdims[1]))
        max_amplitude = np.zeros((simulation.nkdims[0], simulation.nkdims[1]))
        
        # Loop over all modes
        for ikx in range(simulation.nkdims[0]):
            for iky in range(simulation.nkdims[1]):
                # Extract mode amplitude over time
                mode_data = field[:, ikx, iky, z_idx]
                amplitude = np.abs(mode_data)
                
                # Store maximum amplitude
                max_amplitude[ikx, iky] = np.max(amplitude)
                
                # Skip modes with very small amplitude to avoid numerical issues
                if max_amplitude[ikx, iky] < 1e-10:
                    growth_rates[ikx, iky] = 0.0
                    continue
                
                # Compute growth rate using linear fit in log space
                if np.all(amplitude > 0):  # Ensure all amplitudes are positive
                    try:
                        # Use only the growing part for the fit
                        # Take log of amplitude for linear fit
                        log_amplitude = np.log(amplitude)
                        
                        # Linear fit: log(A) = γt + c
                        # Fit a line to the log of the amplitude
                        polyfit = np.polyfit(time, log_amplitude, 1)
                        growth_rates[ikx, iky] = polyfit[0]  # Slope = growth rate
                    except:
                        growth_rates[ikx, iky] = 0.0
                else:
                    growth_rates[ikx, iky] = 0.0
        
        return growth_rates, max_amplitude
