# main.py
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:34:17 2025

@author: Cormac Molyneaux
"""
# main.py
from msd_system import MSDSystem
from data_generator import generate_pristine_data
from analysis import run_monte_carlo_sweep, summarize_results, plot_time_series, plot_histograms
from regression_models import OLSModel, ODRModel
import config
import argparse

def main():
    ask_params = config.SIMULATION_PARAMS.get('ask_params')
    if ask_params:
        parser = argparse.ArgumentParser(description="Run ODE parameter recovery simulations.")
    
        # Arguments for SYSTEM_PARAMS
        parser.add_argument('--m', type=float, help='Mass parameter (m). Overrides config.SYSTEM_PARAMS["m"]')
        parser.add_argument('--c', type=float, help='Damping coefficient (c). Overrides config.SYSTEM_PARAMS["c"]')
        parser.add_argument('--k', type=float, help='Spring constant (k). Overrides config.SYSTEM_PARAMS["k"]')
    
        # Argument for SIMULATION_PARAMS
        parser.add_argument('--file_name', type=str, help='Output file name for results. Overrides config.SIMULATION_PARAMS["file_name"]')
    
        # Arguments for NOISE_CONFIG.filter
        parser.add_argument('--filter_type', type=str,
                            help='Type of filter (e.g., low_pass, savitzky_golay). Overrides config.NOISE_CONFIG["filter"]["type"]')
    
        # Argument for REGRESSION_CONFIG
        parser.add_argument('--regression_method', type=str,
                            help='Regression method (e.g., OLS, ODR, KF). Overrides config.REGRESSION_CONFIG["regression_method"]')
    
        args = parser.parse_args()
    
        # Apply command-line overrides to the config dictionary
        if args.m is not None:
            config.SYSTEM_PARAMS['m'] = args.m
        if args.c is not None:
            config.SYSTEM_PARAMS['c'] = args.c
        if args.k is not None:
            config.SYSTEM_PARAMS['k'] = args.k
        if args.file_name is not None:
            config.SIMULATION_PARAMS['file_name'] = args.file_name
        if args.filter_type is not None:
            config.NOISE_CONFIG['filter']['type'] = args.filter_type
        if args.regression_method is not None:
            config.REGRESSION_CONFIG['regression_method'] = args.regression_method
            
    # 1. Setup System
    msd_system = MSDSystem(**config.SYSTEM_PARAMS)

    # 2. Select Regression Model
    regression_model = None
    if config.REGRESSION_CONFIG.get('regression_method') == 'OLS':
        regression_model = OLSModel()
    elif  config.REGRESSION_CONFIG.get('regression_method') == 'ODR':
        regression_model = ODRModel() 
    else:
        raise ValueError(f"Unknown REGRESSION_METHOD: {config.REGRESSION_METHOD}")

    # 3. Run Monte Carlo Simulation
    results = run_monte_carlo_sweep(msd_system, config.SIMULATION_PARAMS, regression_model, config.REGRESSION_CONFIG, config.NOISE_CONFIG)

    # 4. Summarize Results
    summarize_results(msd_system, config.SIMULATION_PARAMS, \
                              results, msd_system.c, msd_system.k, msd_system.m)

    # 5. Plotting Results 
    last_m_hat = results['m_hats'][-1] 
    last_c_hat = results['c_hats'][-1]
    last_k_hat = results['k_hats'][-1]

    # Create a system with the estimated parameters for prediction plotting
    msd_system_estimated = MSDSystem(m=last_m_hat, c=last_c_hat, k=last_k_hat,
                                    x0=msd_system.x0, v0=msd_system.v0,
                                    force_amplitude=msd_system.force_amplitude,
                                    force_frequency=msd_system.force_frequency)

    x_pred_plot, v_pred_plot, a_pred_plot, _, _ = \
        generate_pristine_data(msd_system_estimated, config.SIMULATION_PARAMS['num_steps'],
                               config.SIMULATION_PARAMS['time_increment'])

    plot_time_series(results['x_pristine'], results['last_x_noisy'], results['last_x_filtered'], x_pred_plot, results['time_points'],
                     'True, Noisy, and Predicted Position', 'Position (m)', 'Position', config.SYSTEM_PARAMS)
    if config.REGRESSION_CONFIG.get('regression_method') != 'KF':
        plot_time_series(results['v_pristine'], results['last_v_noisy'], None, v_pred_plot, results['time_points'],
                         'True, Noisy, and Predicted Velocity', 'Velocity (m/s)', 'Velocity', config.SYSTEM_PARAMS)
        plot_time_series(results['a_pristine'], results['last_a_noisy'], None, a_pred_plot, results['time_points'],
                         'True, Noisy, and Predicted Acceleration', 'Acceleration (m/s^2)', 'Acceleration', config.SYSTEM_PARAMS)
    plot_time_series(results['f_pristine'], results['last_f_noisy'], None, None, results['time_points'],
                     'True and Noisy Force', 'Force (N)', 'Force', config.SYSTEM_PARAMS)

    # Plot histograms of estimated m, c and k values
    plot_histograms(results['c_hats'], results['k_hats'], results['m_hats'], msd_system.c, msd_system.k,  msd_system.m)

if __name__ == "__main__":
    main()