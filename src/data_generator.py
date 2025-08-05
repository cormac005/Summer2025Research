# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:30:01 2025

@author: Cormac Molyneaux
"""
# data_generator.py
import numpy as np
from scipy.integrate import odeint
from msd_system import MSDSystem
#import matplotlib.pyplot as plt
# Placeholder for signal processing functions (will be in signal_processing.py)
# This separation is key for modularity.
from optimiser import optimise_spline_hyperparameters, low_pass_optimiser
from signal_processing import differentiate_data_numerically, apply_savitzky_golay_filter,\
                                apply_spline_differentiation, apply_butterworth_filter, \
                                    fit_to_underdamped_x, apply_kalman_filter, apply_gp_differentiation

def generate_pristine_data(msd_system: MSDSystem, num_steps: int, time_increment: float):
    time_points = np.linspace(0, num_steps * time_increment, num_steps)
    initial_state = [msd_system.x0, msd_system.v0]
    
    # Integrate the system dynamics
    sol = odeint(msd_system.get_continuous_dynamics, initial_state, time_points, args=(msd_system,)) 
    
    x_pristine = sol[:, 0]
    v_pristine = sol[:, 1]
    
    
    f_pristine = msd_system.force_amplitude * np.cos(msd_system.force_frequency * time_points)
    
    # Calculate pristine acceleration from the model
    a_pristine = np.array([msd_system.get_acceleration(x_p, v_p, t_p) 
                           for x_p, v_p, t_p in zip(x_pristine, v_pristine, time_points)])
    
    return x_pristine, v_pristine, a_pristine, f_pristine, time_points

def add_noise_to_data(msd_system: MSDSystem, x_pristine, v_pristine, a_pristine, \
                      f_pristine, noise_config, time_increment, num_steps):
    noise_type = noise_config.get('type', 'independent_xva')
    noise_level_x = noise_config.get('level_x', 0.0)
    noise_level_v = noise_config.get('level_v', 0.0)
    noise_level_a = noise_config.get('level_a', 0.0)
    noise_level_f = noise_config.get('level_f', 0.0)
    f_noise_method = noise_config.get('f_noise_method', 'added_to_true_f')

    #np.random.seed(noise_config.get('seed')) # For reproducibility

    std_dev_x = noise_level_x * np.max(np.abs(x_pristine)) if np.max(np.abs(x_pristine)) > 1e-9 else noise_level_x
    std_dev_v = noise_level_v * np.max(np.abs(v_pristine)) if np.max(np.abs(v_pristine)) > 1e-9 else noise_level_v
    std_dev_a = noise_level_a * np.max(np.abs(a_pristine)) if np.max(np.abs(a_pristine)) > 1e-9 else noise_level_a
    std_dev_f = noise_level_f * np.max(np.abs(f_pristine)) if np.max(np.abs(f_pristine)) > 1e-9 else noise_level_f

    # Ensure minimum non-zero std dev for very small signals
    min_std_dev = 1e-9
    std_dev_x = max(std_dev_x, min_std_dev)
    std_dev_v = max(std_dev_v, min_std_dev)
    std_dev_a = max(std_dev_a, min_std_dev)
    std_dev_f = max(std_dev_f, min_std_dev)

    x_noisy, v_noisy, a_noisy, f_noisy = np.copy(x_pristine), np.copy(v_pristine), np.copy(a_pristine), np.copy(f_pristine)

    if noise_type == 'independent_xva':
        x_noisy += np.random.normal(0, std_dev_x, x_pristine.shape)
        v_noisy += np.random.normal(0, std_dev_v, v_pristine.shape)
        a_noisy += np.random.normal(0, std_dev_a, a_pristine.shape)
        
        # More reliable f_noisy calc using pristine parameters and noisy x, v, a
        if f_noise_method == 'added_to_true_f':
            f_noisy += np.random.normal(0, std_dev_f, f_pristine.shape)
        elif f_noise_method == 'derived_from_noisy_states':
            f = msd_system.m * a_noisy + msd_system.c * v_noisy + msd_system.k * x_noisy 
            f_noisy = f + np.random.normal(0, std_dev_f, f_pristine.shape)
        else:
            raise ValueError(f"Unknown f_noise_method: {f_noise_method}. Choose 'added_to_true_f' or 'derived_from_noisy_states'.")

        # The actual std dev of noise added is what we calculated
        return x_noisy, f_noisy, v_noisy, a_noisy, std_dev_x, std_dev_f, std_dev_v, std_dev_a

    elif noise_type == 'differentiated_from_x':
        # Add noise to x and f
        x_noisy += np.random.normal(0, std_dev_x, x_pristine.shape)
        f_noisy += np.random.normal(0, std_dev_f, f_pristine.shape)

        # Returns the base noise levels for clarity, likely unsuitable for ODR.
        return x_noisy, f_noisy, None, None, std_dev_x, std_dev_f, None, None

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
        
def apply_filtering(x_noisy, f_noisy, noise_config, time_increment, num_steps, std_dev_x, std_dev_f):
    
    # Get Time Points
    time_points = np.linspace(0, (len(x_noisy) - 1) * time_increment, len(x_noisy))
    
    # Apply filter if specified in config (e.g., for SG filtering)
    filter_config = noise_config.get('filter', None)['type']
    if filter_config == 'spline':
        spline_config = noise_config.get('filter', {}).get('spline_config', {}) 
        s = spline_config.get('s', None)
        k = spline_config.get('k', 3)
        
        if s == None:
            k, s = optimise_spline_hyperparameters(x_noisy, time_points)
            #print(f'Chosen k = {k}, Chosen s = {s}')
            
        x_filtered, v_filtered, a_filtered = apply_spline_differentiation(x_noisy, time_points, k, s)
        
    elif filter_config == 'low_pass':
        l_p_config = noise_config.get('filter', {}).get('low_pass_config', {}) 
        cutoff_frequency = l_p_config.get('cutoff_frequency', None)
        order = l_p_config.get('order', None)
        sampling_frequency = 1.0 / time_increment
        
        if cutoff_frequency == None or order == None:
            cutoff_frequency, order = low_pass_optimiser(x_noisy, time_points, \
                                                          time_increment, sampling_frequency)
            
        # Apply low-pass filter to x_noisy
        x_filtered = apply_butterworth_filter(x_noisy, cutoff_frequency, sampling_frequency, order)

        # Numerically differentiate the filtered x to get v and a
        v_filtered, a_filtered = differentiate_data_numerically(x_filtered, time_increment)
    
    elif filter_config == 'gaussian_proccess':
    
        x_filtered, v_filtered, a_filtered, std_dev_x = apply_gp_differentiation(x_noisy, time_points)
            
    
    else:
        x_filtered = []
        # Numerically differentiate to get v and a
        v_noisy, a_noisy = differentiate_data_numerically(x_noisy, time_increment)
    # This return is only for when se_v and se_a have not been calculated
    return x_filtered, v_filtered, a_filtered, std_dev_x, std_dev_x*5, std_dev_x*20