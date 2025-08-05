# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:34:14 2025

@author: Cormac Molyneaux
""" 
# signal_processing.py
import numpy as np
from pykalman import KalmanFilter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.signal import butter, filtfilt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def apply_butterworth_filter(data, cutoff_frequency, sampling_frequency, order=4):
    """Applies a Butterworth low-pass filter (zero-phase)."""
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def differentiate_data_numerically(x_data, dt):
    """
    Numerically differentiates position to get velocity and acceleration.
    Uses np.gradient for robust differentiation, handling endpoints.
    """
    v_data = np.gradient(x_data, dt)
    a_data = np.gradient(v_data, dt)
    return v_data, a_data

def apply_spline_differentiation(x_data, time_points, k=3, s=None):
    """
    Fits a spline to position data and differentiates it to get velocity and acceleration.

    Parameters:
        x_data (np.array): The noisy position data.
        time_increment (np.array): The corresponding time points.
        k (int): Degree of the spline (default is 3 for cubic spline).
                 Higher degrees can fit more complex curves but may be less stable.
        s (float, optional): Positive smoothing factor used to choose the number of knots.
                             Default (None) means that `s` will be `len(x_data) - sqrt(2*len(x_data))`,
                             which usually provides a good fit for interpolation.
                             If `s=0`, the spline interpolates all points.
                             A larger `s` results in a smoother spline (more deviation from data).
                             This is crucial for noise reduction.

    Returns:
        tuple: (x_spline, v_spline, a_spline)
               - x_spline (np.array): Smoothed position from the spline.
               - v_spline (np.array): Velocity obtained by differentiating the spline.
               - a_spline (np.array): Acceleration obtained by twice differentiating the spline.
    """
    if len(time_points) != len(x_data):
        raise ValueError("time_points and x_data must have the same length.")
    if len(x_data) < k + 1:
        # Need at least k+1 points to fit a spline of degree k
        print(f"Warning: Not enough data points ({len(x_data)}) to fit a spline of degree {k}. Returning NaNs.")
        return np.full_like(x_data, np.nan), np.full_like(x_data, np.nan), np.full_like(x_data, np.nan)
        
    # Step 1: Create the BSpline object
    # make_interp_spline returns a BSpline object directly
    spline_obj = UnivariateSpline(time_points, x_data, k=k, s=s)

    # Step 2: Evaluate the spline for position
    x_spline = spline_obj(time_points)

    # Step 3: Differentiate the spline to get velocity (1st derivative)
    v_spline_obj = spline_obj.derivative(1)
    v_spline = v_spline_obj(time_points)

    # Step 4: Differentiate again to get acceleration (2nd derivative)
    a_spline_obj = spline_obj.derivative(2)
    a_spline = a_spline_obj(time_points)

    return x_spline, v_spline, a_spline

def under_damped_model(t, C, s, w, phi):
    """
    The non-linear model function for x(t) = exp(st) * cos(wt + phi)
    """
    return C * np.exp(s * t) * np.cos(w * t + phi)

def under_damped_model_velocity(t, C, s, w, phi):
    return C * np.exp(s * t) * (s * np.cos(w * t + phi) - w * np.sin(w * t + phi))

def under_damped_model_acceleration(t, C, s, w, phi):
    return C * np.exp(s * t) * ((s**2 - w**2) * np.cos(w * t + phi) - 2 * w * s * np.sin(w * t + phi))

def fit_to_underdamped_x(x_data, time_increment, num_steps):
    #Set initial guesses for C, s, w and phi
    initial_guess = [1.0, -1.0, 2.0, 0.0]
    
    # Calculate the end time
    end_time = (num_steps - 1) * time_increment
    
    # Define the time_points array
    time_points = np.linspace(0, end_time, num_steps)
    
    #Run regression
    popt, pcov = curve_fit(under_damped_model, time_points, x_data, p0=initial_guess)
    
    #Pull data out
    C_hat, s_hat, w_hat, phi_hat = popt
    perr = np.sqrt(np.diag(pcov))
    SE_C, SE_s, SE_w, SE_phi = perr
    
    x_derived = []
    v_derived = []
    a_derived = []
    
    #Calculate x, v and a using guesses
    x_derived = under_damped_model(time_points, C_hat, s_hat, w_hat, phi_hat)
    v_derived = under_damped_model_velocity(time_points, C_hat, s_hat, w_hat, phi_hat)
    a_derived = under_damped_model_acceleration(time_points, C_hat, s_hat, w_hat, phi_hat)
        
    return x_derived, v_derived, a_derived, s_hat, w_hat, phi_hat, SE_s, SE_w, SE_phi


def apply_gp_differentiation(x_data, time_points):
    """
    Fits a Gaussian Process model to the data and calculates its derivatives.
    """
    # Reshape time_points for sklearn
    t = time_points.reshape(-1, 1)

    # Define the GP kernel
    # An RBF kernel is a good starting point for smooth functions
    # A WhiteKernel accounts for the noise in the observations
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
             WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6 , 1e5))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10)

    # Fit the GP to the noisy position data
    gp.fit(t, x_data)

    # Predict the smoothed position and its standard deviation
    x_gp, std_x_gp = gp.predict(t, return_std=True)
    v_gp = np.gradient(x_gp, time_points)
    a_gp = np.gradient(v_gp, time_points)

    return x_gp, v_gp, a_gp, std_x_gp