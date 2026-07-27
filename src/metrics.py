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
        if window_size <= 0:
            return np.asarray(trace, dtype=float).copy()
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
        """Backward-compatible access to the manuscript's total NRMSE.

        The reference is always the Monte Carlo trace.  This convention is also
        used by :meth:`compare_activity`, which is the canonical implementation
        for every table and figure in the corrected revision.
        """
        metrics = ActivityMetrics.compare_activity(
            A_mc, A_pde, dt, smooth_window=smooth_window
        )
        return metrics["nrmse"], metrics["rmse_hz"]

    @staticmethod
    def compare_activity(A_mc, A_cl, dt, smooth_window=0.05):
        """Return a single, explicit set of MC-versus-closure metrics.

        ``bias_hz`` follows the manuscript definition ``mean(A_cl - A_mc)``.
        Bias and RMSE are in Hz; NRMSE values are dimensionless and use the
        temporal standard deviation of the smoothed MC reference.  The total
        and centered errors are both reported instead of silently swapping one
        for the other in different analyses.
        """
        A_mc = np.asarray(A_mc, dtype=float)
        A_cl = np.asarray(A_cl, dtype=float)
        length = min(A_mc.size, A_cl.size)
        if length == 0:
            raise ValueError("activity traces must be non-empty")
        mc_sm = ActivityMetrics.smooth_trace(A_mc[:length], dt, smooth_window)
        cl_sm = ActivityMetrics.smooth_trace(A_cl[:length], dt, smooth_window)
        diff = cl_sm - mc_sm
        bias = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        crmse = float(np.sqrt(np.mean((diff - bias) ** 2)))
        mc_std = float(np.std(mc_sm))
        denom = max(mc_std, 1e-12)
        return {
            "bias_hz": bias,
            "rmse_hz": rmse,
            "crmse_hz": crmse,
            "mc_std_hz": mc_std,
            "nrmse": rmse / denom,
            "centered_nrmse": crmse / denom,
            "mc_mean_hz": float(np.mean(mc_sm)),
            "closure_mean_hz": float(np.mean(cl_sm)),
        }
