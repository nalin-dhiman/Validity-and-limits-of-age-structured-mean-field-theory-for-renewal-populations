import numpy as np
from scipy.ndimage import gaussian_filter1d

class ActivityMetrics:
    @staticmethod
    def bin_spikes(spike_times, N, T, dt_bin):
        pass
        
    @staticmethod
    def compute_rate(count_trace, N, bin_width):
        """
        Convert binned spike counts to rate (Hz).
        """
        return count_trace / (N * bin_width)
    
    @staticmethod
    def smooth_trace(trace, dt, window_size=0.02, method='gaussian'):
        """
        Smooth activity trace.
        window_size: seconds (sigma for gaussian, tau for exponential)
        """
        if method == 'gaussian':
            sigma_steps = window_size / dt
            return gaussian_filter1d(trace, sigma_steps)
        elif method == 'exponential':
            tau = window_size
            length = int(5 * tau / dt)
            t_kernel = np.arange(length) * dt
            # Kernel K(t) = (1/tau) * exp(-t/tau)
            kernel = (1.0 / tau) * np.exp(-t_kernel / tau)
            # Normalize to sum=1 to preserve mean rate
            kernel = kernel / np.sum(kernel)
            # Convolve
            # mode='full' then crop? or 'same'?
            # Causal usually implies we care about t-s.
            # strict convolution is causal if we shift?
            # np.convolve(a, v, mode='full') returns len(a)+len(v)-1.
            # We want output same size.
            # To be strictly causal (only past affects future), 
            # we should align the end of kernel with current point?
            # Standard "smooth" usually centers the window?
            # User asked for "Exponential Kernel ... K(t)=... 1_{t>=0}". This is causal.
            # So y(t) depends on x(s) where s <= t.
            # So convolution $y = x * k$.
            y = np.convolve(trace, kernel, mode='full')
            return y[:len(trace)] # Truncate to original length (causal delay included)
        else:
            return trace

    @staticmethod
    def compute_nrmse(A_mc, A_pde, dt, smooth_window=0.05):
        """
        Compute Normalized RMSE on smoothed traces.
        """
        A1 = ActivityMetrics.smooth_trace(A_mc, dt, smooth_window)
        A2 = ActivityMetrics.smooth_trace(A_pde, dt, smooth_window)
        
        # Crop artifacts? 
        # User said "burn-in 10s". The traces passed here should be already cropped.
        # But if not, we assume inputs are aligned.
        
        diff = A1 - A2
        rmse = np.sqrt(np.mean(diff**2))
        normalization = np.std(A1) + 1e-12
        
        return rmse / normalization, rmse
