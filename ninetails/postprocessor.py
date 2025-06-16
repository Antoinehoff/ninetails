# post_processing.py
import numpy as np
from .tools import slice_average

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

    def compute_growth_rates(simulation, time, field, tlim=[], z_idx=0, return_error=False):
        # Extract time range if tlim is provided
        if tlim:
            time_indices = np.where((time >= tlim[0]) & (time <= tlim[1]))[0]
        else:
            time_indices = np.arange(len(time) // 2)  # Default: use half of the time range
        time = time[time_indices]
        field = field[time_indices, :, :, :]

        # Initialize growth rate arrays
        growth_rates = np.zeros((simulation.nkdims[0], simulation.nkdims[1]))
        errors = np.zeros((simulation.nkdims[0], simulation.nkdims[1]))
        total_err = 0.0

        # Loop over all modes
        for ikx in range(simulation.nkdims[0]):
            for iky in range(simulation.nkdims[1]):
                # Extract mode amplitude over time
                mode_data = np.abs(field[:, ikx, iky, z_idx])

                # Skip modes with very small amplitude to avoid numerical issues
                if np.max(mode_data) < 1e-10:
                    continue

                # Compute growth rates using corrected method
                gamma = np.zeros(len(time))
                for it in range(1, len(time)):
                    y_n = mode_data[it]
                    y_nm1 = mode_data[it - 1]
                    dt = time[it] - time[it - 1]
                    
                    # Avoid log of zero or negative values
                    if y_n > 0 and y_nm1 > 0:
                        gamma[it] = np.log(y_n / y_nm1) / dt
                    else:
                        gamma[it] = 0.0

                # Error estimation (slice averaging)
                n = min(5, len(gamma) // 2)  # Number of points for averaging
                if n > 1:
                    gamma_avg, _, gamma_err = slice_average(gamma, n)
                    growth_rates[ikx, iky] = np.mean(gamma_avg[n:])  # Skip initial transient
                    errors[ikx, iky] = np.mean(gamma_err[n:])
                    total_err += errors[ikx, iky]
                else:
                    growth_rates[ikx, iky] = np.mean(gamma[len(gamma)//2:])  # Use second half
                    errors[ikx, iky] = np.std(gamma[len(gamma)//2:])
                    total_err += errors[ikx, iky]

        # Normalize total error by the number of modes
        total_err /= (simulation.nkdims[0] * simulation.nkdims[1])

        if return_error:
            return growth_rates, total_err, errors
        else:
            return growth_rates, total_err

    def compute_mode_amplitude_evolution(simulation, time, field, z_idx=0):
        mode_amplitudes_kx = np.abs(field[:, :, 0, z_idx])  # ky=0
        mode_amplitudes_ky = np.abs(field[:, 0, :, z_idx])  # kx=0
        return mode_amplitudes_kx, mode_amplitudes_ky
