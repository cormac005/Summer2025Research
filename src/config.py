# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:34:18 2025

@author: Cormac Molyneaux
"""
# config.py
# Centralized configuration for the simulation and analysis

SYSTEM_PARAMS = {
    'm': 2.0, 
    'c': 2.0,
    'k': 9.0,
    'x0': 0.0,
    'v0': 0.0,
    'force_amplitude': 1.0, # Set to 0.0 for unforced
    'force_frequency': 5.0 # Relevant only if force_amplitude = 0
}

SIMULATION_PARAMS = {
    'num_simulations': 100,
    'num_steps': 1000,
    'time_increment': 0.005,
    'diagnostic_statistics': False, # True or False
    'save_results': True, #True or False
    'file_name': 'simulation_results_low_pass.pkl', #Only matters if save_results is True
    'ask_params': True #True or False, set to true when running from run_big_sim.py
}

# Noise configuration. Choose one 'type'.
# 'independent_xva': Adds independent noise to pristine x, v, a.
# 'differentiated_from_x': Adds noise only to pristine x, then differentiates x_noisy.
NOISE_CONFIG = {
    'type': 'differentiated_from_x', # 'independent_xva' or 'differentiated_from_x'
    'level_x': 0.01,  #Noise percentage (std dev / max_abs_value)
    'level_v': 0.075,
    'level_a': 0.1,
    'level_f': 0.01, 
    'seed': 42, # For reproducibility of noise
    #If 'independent_xva' is chosen, choose how f_noisy is generated
    # 'added_to_true_f': f_noisy = f_pristine + Noise_f (Noise_f is independent) (works with ODR)
    # 'derived_from_noisy_states': f_noisy = m*a_noisy + c*v_noisy + k*x_noisy + Noise_f (works with OLS)
    'f_noise_method': 'added_to_true_f', #'added_to_true_f' or 'derived_from_noisy_states'
    
    # If 'differentiated_from_x' is chosen, you can add filter config:
    # Note: if any hyper parameters are set to None, they will be optimised
    'filter': {
        'type': 'spline', #'low_pass', 'spline' or 'gaussian_proccess'
        # Parameters for spline filter
        'spline_config': {
            'k': None,  # Cubic spline degree
            's': None, # Set to None to optimise
        },
        # Parameters for low-pass filter
        'low_pass_config': { 
            'cutoff_frequency': None, # Hz. Tune this based on your signal's frequency content.
            'order': None,
        },
        }
}

# Which regression model to use for the sweep
REGRESSION_CONFIG = {
    'regression_method': 'ODR', # 'OLS' or 'ODR'
    'amount_to_truncate_start': 100, #num data points removed from the start before regression
    'amount_to_truncate_end': 100, #num data points removed from the end before regression
    }
