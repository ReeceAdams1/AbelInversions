import numpy as np
import abel
import abel.basex
import abel.hansenlaw
import abel.dasch
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

def find_center_of_mass(data):
    """
    Finds the center of mass of the given 1D profile.
    """
    # Remove NaNs
    valid_mask = np.isfinite(data)
    data_clean = data[valid_mask]
    x_vals = np.arange(len(data))[valid_mask]
    
    if len(data_clean) == 0:
        return None

    try:
        bg = np.min(data_clean)
        weights = data_clean - bg
        weights = np.maximum(weights, 0) # Ensure non-negative
        if np.sum(weights) > 0:
            com = np.average(x_vals, weights=weights)
            return int(round(com))
    except Exception as e:
        print(f"CoM failed: {e}")
    return None

def find_center_gaussian(data, center_guess=None):
    """
    Finds the center by fitting a Gaussian to the profile.
    """
    # Remove NaNs
    valid_mask = np.isfinite(data)
    data_clean = data[valid_mask]
    x_vals = np.arange(len(data))[valid_mask]
    
    if len(data_clean) == 0:
        return None

    def gaussian(x, amp, cen, wid, bg):
        return amp * np.exp(-(x-cen)**2 / (2*wid**2)) + bg
        
    try:
        amp_guess = np.max(data_clean) - np.min(data_clean)
        cen_guess = center_guess if center_guess is not None else np.mean(x_vals)
        wid_guess = len(data_clean) / 4
        bg_guess = np.min(data_clean)
        
        p0 = [amp_guess, cen_guess, wid_guess, bg_guess]
        popt, _ = curve_fit(gaussian, x_vals, data_clean, p0=p0, maxfev=5000)
        return int(round(popt[1]))
    except Exception as e:
        print(f"Gaussian fit failed: {e}")
    return None

def detect_widths(data, center):
    """
    Auto-detects the left and right widths of the signal based on noise floor.
    Returns (left_width, right_width, threshold, noise_stats_msg).
    """
    # Handle NaNs
    data = np.nan_to_num(data, nan=0.0)
    n_pts = len(data)
    peak_val = np.max(data)
    
    # Determine noise floor
    # Use 5% of points at edges
    edge_pts = int(n_pts * 0.05)
    if edge_pts < 5: edge_pts = 5
    
    left_edge = data[:edge_pts]
    right_edge = data[-edge_pts:]
    
    # Calculate stats for both edges separately
    l_mean, l_std = np.mean(left_edge), np.std(left_edge)
    r_mean, r_std = np.mean(right_edge), np.std(right_edge)
    
    # Use the side with lower noise (std dev) to avoid including signal in noise estimate
    if l_std < r_std:
        noise_mean = l_mean
        noise_std = l_std
        used_edge = "Left"
    else:
        noise_mean = r_mean
        noise_std = r_std
        used_edge = "Right"
    
    # Threshold strategy:
    # 1. 3-sigma above noise floor
    # 2. 1% of peak amplitude (relative to noise floor)
    threshold_sigma = noise_mean + 3.0 * noise_std
    threshold_pct = noise_mean + 0.01 * (peak_val - noise_mean)
    
    threshold = max(threshold_sigma, threshold_pct)
    
    # Sanity check: Ensure threshold is not above the peak (or too close)
    if threshold >= peak_val * 0.95:
        threshold = noise_mean + 0.1 * (peak_val - noise_mean)
    
    # Smooth data slightly for detection
    try:
        data_smooth = savgol_filter(data, 11, 3)
    except:
        data_smooth = data

    # Find Left Cutoff
    left_idx = 0
    for i in range(center, -1, -1):
        if data_smooth[i] < threshold:
            left_idx = i
            break
    
    # Find Right Cutoff
    right_idx = n_pts - 1
    for i in range(center, n_pts):
        if data_smooth[i] < threshold:
            right_idx = i
            break
    
    msg = (f"Detected cutoffs:\nLeft: {left_idx}\nRight: {right_idx}\n"
           f"Threshold: {threshold:.2e}\n"
           f"Noise Source: {used_edge} Edge (Mean: {noise_mean:.2e}, Std: {noise_std:.2e})")
           
    return left_idx, right_idx, threshold, msg

def taper_profile(profile):
    """
    Tapers the edge of the profile to zero using an exponential decay
    matched to the slope of the last 20 points.
    """
    if len(profile) == 0: return profile
    
    # Check last value
    last_val = profile[-1]
    if last_val <= 0: return profile 
    
    # Calculate slope of last 20 points
    n_slope = min(20, len(profile))
    if n_slope < 2:
        slope = 0
    else:
        y_tail = profile[-n_slope:]
        x_tail = np.arange(n_slope)
        # Fit line: y = mx + c
        slope, _ = np.polyfit(x_tail, y_tail, 1)
    
    # Match slope at x=0 for f(x) = A * exp(-k*x)
    # f(0) = A = last_val
    # f'(0) = -A * k = slope  =>  k = -slope / last_val
    
    if slope >= 0:
        # Fallback for rising/flat edge: Decay to 1% over 20 pixels
        k = 4.6 / 20.0 
    else:
        k = -slope / last_val
        
    # Ensure k isn't too small (limit max length)
    if k < 0.007: k = 0.007 # Limit to ~1000 pixels length
        
    # Determine length to drop to 0.1%
    taper_len = int(np.ceil(6.9 / k))
    if taper_len > 1000: taper_len = 1000
        
    x = np.arange(1, taper_len + 1)
    taper = last_val * np.exp(-k * x)
    
    return np.concatenate([profile, taper])

def subtract_baseline(profile):
    """
    Subtracts a constant baseline calculated from the tail (last 10% of points).
    """
    if len(profile) < 10: return profile
    
    n_tail = max(5, int(len(profile) * 0.1))
    baseline = np.mean(profile[-n_tail:])
    
    return profile - baseline

def prepare_profiles(data, center, left_width, right_width, 
                    smooth_l_params, smooth_r_params, 
                    smoothing_method='savgol',
                    use_cutoff=False, taper_edges=False, subtract_bg=False):
    """
    Extracts, smooths, and prepares left and right profiles for inversion.
    smooth_params: (window, poly)
    smoothing_method: 'savgol', 'gaussian', or 'none'
    """
    data_analysis = np.nan_to_num(data, nan=0.0)
    
    sw_l, sp_l = smooth_l_params
    sw_r, sp_r = smooth_r_params
    
    # Ensure odd window length for savgol
    if sw_l % 2 == 0: sw_l += 1
    if sw_r % 2 == 0: sw_r += 1

    # Determine widths
    if use_cutoff:
        l_width = left_width
        r_width = right_width
    else:
        l_width = center
        r_width = len(data_analysis) - center

    # Left (Include center pixel to ensure r=0 is correct)
    left_raw = data_analysis[max(0, center-l_width):center+1][::-1]
    
    # Right (Include center pixel)
    right_raw = data_analysis[center:min(len(data_analysis), center+r_width+1)]

    # Subtract Background if requested (before smoothing)
    if subtract_bg:
        left_raw = subtract_baseline(left_raw)
        right_raw = subtract_baseline(right_raw)
        # Clip negative values to 0 to prevent artifacts
        left_raw = np.maximum(left_raw, 0)
        right_raw = np.maximum(right_raw, 0)

    # Smoothing
    if smoothing_method == 'none':
        left_smooth = left_raw
        right_smooth = right_raw
    elif smoothing_method == 'gaussian':
        # Interpret window as sigma (scaled down)
        # Window 11 -> Sigma 1.1
        sigma_l = sw_l / 10.0
        sigma_r = sw_r / 10.0
        left_smooth = gaussian_filter1d(left_raw, sigma_l)
        right_smooth = gaussian_filter1d(right_raw, sigma_r)
    else: # savgol (default)
        if len(left_raw) > sw_l:
            left_smooth = savgol_filter(left_raw, sw_l, sp_l)
        else:
            left_smooth = left_raw

        if len(right_raw) > sw_r:
            right_smooth = savgol_filter(right_raw, sw_r, sp_r)
        else:
            right_smooth = right_raw
            
    # Apply Tapering if enabled
    if taper_edges:
        left_smooth = taper_profile(left_smooth)
        right_smooth = taper_profile(right_smooth)
        
    return left_smooth, right_smooth

def perform_inversion(profile, method, pixel_size, direction='inverse'):
    """
    Performs the Abel inversion (or forward transform) on a single profile.
    """
    if method == 'basex':
        # Enable correction for better center accuracy
        return abel.basex.basex_transform(profile, direction=direction, dr=pixel_size, correction=True)
    elif method == 'hansenlaw':
        return abel.hansenlaw.hansenlaw_transform(profile, direction=direction, dr=pixel_size)
    else:
        # Dasch methods
        func = getattr(abel.dasch, f"{method}_transform")
        return func(profile, direction=direction, dr=pixel_size)

def calculate_fwhm(y, x):
    half_max = np.max(y) / 2.0
    indices = np.where(y > half_max)[0]
    if len(indices) > 0:
        return x[indices[-1]] - x[indices[0]]
    return 0.0
