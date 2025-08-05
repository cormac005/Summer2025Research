# analysis.py
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:34:16 2025

@author: Cormac Molyneaux
"""
# analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from data_generator import generate_pristine_data, add_noise_to_data, apply_filtering, differentiate_data_numerically
from msd_system import MSDSystem
from regression_models import OLSModel, ODRModel, KalmanFilterModel
from scipy import stats
from multiprocessing import Pool, cpu_count

def run_single_simulation(args):
    """
    Worker function to run a single Monte Carlo simulation.
    Takes a tuple of arguments to be compatible with multiprocessing.Pool.
    """
    # 1. Unpack arguments
    sim_index, msd_system, sim_config, regression_model, regression_config, noise_config, \
    x_pristine, v_pristine, a_pristine, f_pristine, time_points = args

    # 2. Important: Seed the random number generator for this specific process
    # This ensures each simulation gets a unique, but reproducible, noise pattern.
    if 'seed' in noise_config:
        np.random.seed(noise_config['seed'] + sim_index)

    # 3. Add noise to the pristine data
    x_noisy, f_noisy, v_noisy, a_noisy, std_dev_x, std_dev_f, std_dev_v, std_dev_a = \
        add_noise_to_data(msd_system, x_pristine, v_pristine, a_pristine, f_pristine, noise_config, sim_config['time_increment'], sim_config['num_steps'])

    # 4. Initialize regression data variables
    x_for_regression, v_for_regression, a_for_regression, f_for_regression = x_noisy, v_noisy, a_noisy, f_noisy
    SE_x_for_regression, SE_v_for_regression, SE_a_for_regression = std_dev_x, std_dev_v, std_dev_a
    x_filtered = None

    # 5. Apply filtering if required
    if noise_config['type'] == 'differentiated_from_x' and not isinstance(regression_model, KalmanFilterModel):
        x_filtered, v_filtered, a_filtered, std_error_x, std_error_v, std_error_a = \
            apply_filtering(x_noisy, f_noisy, noise_config, sim_config['time_increment'], sim_config['num_steps'], std_dev_x, std_dev_f)
        
        x_for_regression, v_for_regression, a_for_regression = x_filtered, v_filtered, a_filtered
        SE_x_for_regression, SE_v_for_regression, SE_a_for_regression = std_error_x, std_error_v, std_error_a
        
        start, end = regression_config.get('amount_to_truncate_start'), regression_config.get('amount_to_truncate_end')
        x_for_regression, v_for_regression, a_for_regression = np.array(x_for_regression[start:-end]), np.array(v_for_regression[start:-end]), np.array(a_for_regression[start:-end])
        f_for_regression = np.array(f_for_regression[start:-end])

        if isinstance(SE_x_for_regression, np.ndarray): SE_x_for_regression = np.array(SE_x_for_regression[start:-end])
        if isinstance(SE_v_for_regression, np.ndarray): SE_v_for_regression = np.array(SE_v_for_regression[start:-end])
        if isinstance(SE_a_for_regression, np.ndarray): SE_a_for_regression = np.array(SE_a_for_regression[start:-end])

    # 6. Run the regression
    df = 0  # Initialize degrees of freedom
    
    if isinstance(regression_model, OLSModel):
        m_hat, c_hat, k_hat, SE_m_hat, SE_c_hat, SE_k_hat, aux_stat, df = \
            regression_model.fit(x_for_regression, v_for_regression, a_for_regression, f_for_regression)
            
    elif isinstance(regression_model, ODRModel):
        m_hat, c_hat, k_hat, SE_m_hat, SE_c_hat, SE_k_hat, aux_stat, df = \
            regression_model.fit(x_for_regression, v_for_regression, a_for_regression, f_for_regression,
                                  std_dev_x=SE_x_for_regression, std_dev_v=SE_v_for_regression,
                                  std_dev_a=SE_a_for_regression, std_dev_f=std_dev_f)
        
    else:
        raise ValueError("Unsupported regression model type.")

    if np.isnan(m_hat) or np.isnan(c_hat) or np.isnan(k_hat):
        return None # Return None for failed runs

    # 7. Calculate confidence intervals and coverage
    t_critical = stats.t.ppf(0.975, df) if df > 0 else 1.96
    m_low, m_high = m_hat - t_critical * SE_m_hat, m_hat + t_critical * SE_m_hat
    c_low, c_high = c_hat - t_critical * SE_c_hat, c_hat + t_critical * SE_c_hat
    k_low, k_high = k_hat - t_critical * SE_k_hat, k_hat + t_critical * SE_k_hat

    m_covered = 1 if m_low <= msd_system.m <= m_high else 0
    c_covered = 1 if c_low <= msd_system.c <= c_high else 0
    k_covered = 1 if k_low <= msd_system.k <= k_high else 0

    # 8. Return all results for this single run in a dictionary
    return {
        'm_hat': m_hat, 'c_hat': c_hat, 'k_hat': k_hat,
        'SE_m_hat': SE_m_hat, 'SE_c_hat': SE_c_hat, 'SE_k_hat': SE_k_hat,
        'm_low': m_low, 'm_high': m_high, 'm_covered': m_covered,
        'c_low': c_low, 'c_high': c_high, 'c_covered': c_covered,
        'k_low': k_low, 'k_high': k_high, 'k_covered': k_covered,
        'aux_stat': aux_stat,
    }

def run_monte_carlo_sweep(msd_system: MSDSystem, sim_config: dict, regression_model, regression_config: dict, noise_config: dict):
    num_simulations = sim_config['num_simulations']
    
    # Generate pristine data once to be shared across all simulations
    x_pristine, v_pristine, a_pristine, f_pristine, time_points = generate_pristine_data(
        msd_system, sim_config['num_steps'], sim_config['time_increment']
    )

    # Prepare arguments for each simulation run
    args_for_pool = [
        (i, msd_system, sim_config, regression_model, regression_config, noise_config,
         x_pristine, v_pristine, a_pristine, f_pristine, time_points)
        for i in range(num_simulations)
    ]
    
    # Run on each CPU core.
    num_processes = cpu_count()
    print(f"Running {num_simulations} simulations in parallel on {num_processes} cores...")

    # Use a Pool of workers to run simulations in parallel
    with Pool(processes=num_processes) as pool:
        # map() sends one argument from the iterable to the worker function.
        parallel_results = pool.map(run_single_simulation, args_for_pool)

    # --- Process the collected results from all workers ---
    
    # Filter out any failed runs (which we returned as None)
    valid_results = [res for res in parallel_results if res is not None]
    
    if not valid_results:
        raise RuntimeError("All simulation runs failed. Check model parameters or noise levels.")

    print(f"Completed {len(valid_results)}/{num_simulations} successful simulations.")

    # Unzip the list of dictionaries into separate lists
    m_hats = [res['m_hat'] for res in valid_results]
    c_hats = [res['c_hat'] for res in valid_results]
    k_hats = [res['k_hat'] for res in valid_results]
    # ... and so on for all other metrics
    m_low_bounds = [res['m_low'] for res in valid_results]
    m_high_bounds = [res['m_high'] for res in valid_results]
    c_low_bounds = [res['c_low'] for res in valid_results]
    c_high_bounds = [res['c_high'] for res in valid_results]
    k_low_bounds = [res['k_low'] for res in valid_results]
    k_high_bounds = [res['k_high'] for res in valid_results]
    SE_m_hats = [res['SE_m_hat'] for res in valid_results]
    SE_c_hats = [res['SE_c_hat'] for res in valid_results]
    SE_k_hats = [res['SE_k_hat'] for res in valid_results]
    aux_regression_stats = [res['aux_stat'] for res in valid_results]
    
    m_covered_count = sum(res['m_covered'] for res in valid_results)
    c_covered_count = sum(res['c_covered'] for res in valid_results)
    k_covered_count = sum(res['k_covered'] for res in valid_results)

    # --- For plotting, generate one final set of noisy data ---
    # This is simpler than trying to capture one from the parallel runs.
    last_x_noisy, last_f_noisy, _, _, std_dev_x, std_dev_f, _, _ = \
        add_noise_to_data(msd_system, x_pristine, v_pristine, a_pristine, f_pristine, noise_config, sim_config['time_increment'], sim_config['num_steps'])
    
    if noise_config['type'] == 'differentiated_from_x':
        last_x_filtered, last_v_noisy, last_a_noisy, _, _, _ = \
            apply_filtering(last_x_noisy, last_f_noisy, noise_config, sim_config['time_increment'], sim_config['num_steps'], std_dev_x, std_dev_f)
    else: # Fallback for independent_xva noise
        last_x_filtered = None
        _, last_v_noisy, last_a_noisy, _ = differentiate_data_numerically(last_x_noisy, sim_config['time_increment'])


    # --- Final calculation for error analysis ---
    if isinstance(regression_model, KalmanFilterModel):
        last_true_error, last_measured_error, bias = None, None, None
    else:
        last_true_error = last_f_noisy - (msd_system.m * last_a_noisy + msd_system.c * last_v_noisy + last_x_filtered)
        last_measured_error = last_f_noisy - (m_hats[-1] * last_a_noisy + c_hats[-1] * last_v_noisy + last_x_filtered)
        bias = last_true_error - last_measured_error
    
    results = {
        'm_hats': np.array(m_hats),
        'c_hats': np.array(c_hats),
        'k_hats': np.array(k_hats),
        'm_low_bounds': np.array(m_low_bounds),
        'm_high_bounds': np.array(m_high_bounds),
        'c_low_bounds': np.array(c_low_bounds),
        'c_high_bounds': np.array(c_high_bounds),
        'k_low_bounds': np.array(k_low_bounds),
        'k_high_bounds': np.array(k_high_bounds),
        'm_covered_count': m_covered_count,
        'c_covered_count': c_covered_count,
        'k_covered_count': k_covered_count,
        'num_simulations': num_simulations,
        'SE_m_hats': np.array(SE_m_hats),
        'SE_c_hats': np.array(SE_c_hats),
        'SE_k_hats': np.array(SE_k_hats),
        'aux_regression_stats': np.array(aux_regression_stats),
        'last_x_noisy': last_x_noisy,
        'last_x_filtered': last_x_filtered,
        'last_v_noisy': last_v_noisy,
        'last_a_noisy': last_a_noisy,
        'last_f_noisy': last_f_noisy,
        'x_pristine': x_pristine,
        'v_pristine': v_pristine,
        'a_pristine': a_pristine,
        'f_pristine': f_pristine,
        'last_true_error': last_true_error,
        'last_measured_error': last_measured_error,
        'bias': bias,
        'time_points': time_points,
        'regression_method': regression_model.__class__.__name__,
        'noise_config': noise_config
    }
    
    if sim_config['save_results']:
        output_filename = sim_config['file_name']
        try:
            with open(output_filename, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results successfully saved to {output_filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
            
    return results

def summarize_results(msd_system: MSDSystem, sim_config: dict, results: dict, \
                      true_c: float, true_k: float, true_m: float):
    num_simulations = results['num_simulations']
    m_hats = results['m_hats']
    c_hats = results['c_hats']
    k_hats = results['k_hats']
    m_covered_count = results['m_covered_count']
    c_covered_count = results['c_covered_count']
    k_covered_count = results['k_covered_count']

    SE_m_hats = results['SE_m_hats']
    SE_c_hats = results['SE_c_hats']
    SE_k_hats = results['SE_k_hats']
    aux_regression_stats = results['aux_regression_stats']
    regression_method = results['regression_method']
    noise_config = results['noise_config']
    
    noise_type = noise_config['type']
    filter_method = noise_config.get('filter', None)
    f_noise_method = noise_config['f_noise_method']

    print("\n--- Simulation Results Summary ---")
    print(f"Regression Method: {regression_method}")
    print(f"Number of Simulations: {num_simulations}")
    
    print(f"\nNoise Case: {noise_type}")
    print(f"Method for adding noise to f: {f_noise_method}")
    # if noise_type == 'differentiated_from_x':
    #     print(f"Filter Used: {filter_method}")

    print(f"\nTrue Parameters: m={true_m:.4f}, c={true_c:.4f}, k={true_k:.4f}")

    print("\nEstimated Mass (m):")
    print(f"  Mean m_hat: {np.mean(m_hats):.4f}")
    print(f"  Std Dev of m_hat (Empirical): {np.std(m_hats):.4f}")
    print(f"  Mean Reported SE(m_hat): {np.mean(SE_m_hats):.4f}")
    m_coverage = (m_covered_count / num_simulations) * 100
    print(f"  m Coverage (95% CI): {m_coverage:.2f}%")

    print("\nEstimated Damping Coefficient (c):")
    print(f"  Mean c_hat: {np.mean(c_hats):.4f}")
    print(f"  Std Dev of c_hat (Empirical): {np.std(c_hats):.4f}")
    print(f"  Mean Reported SE(c_hat): {np.mean(SE_c_hats):.4f}")
    c_coverage = (c_covered_count / num_simulations) * 100
    print(f"  c Coverage (95% CI): {c_coverage:.2f}%")

    print("\nEstimated Stiffness (k):")
    print(f"  Mean k_hat: {np.mean(k_hats):.4f}")
    print(f"  Std Dev of k_hat (Empirical): {np.std(k_hats):.4f}")
    print(f"  Mean Reported SE(k_hat): {np.mean(SE_k_hats):.4f}")
    k_coverage = (k_covered_count / num_simulations) * 100
    print(f"  k Coverage (95% CI): {k_coverage:.2f}%")
    
    print("\nEstimated Governing Equation:")
    print(f" {np.mean(m_hats):.2f}x'' + {np.mean(c_hats):.2f}x' + {np.mean(k_hats):.2f}x = f(t)")

    if sim_config['diagnostic_statistics']: 
        print("\n--- Diagnostic Statistics ---")
        if regression_method == 'ODR':
            mean_res_var = np.mean(aux_regression_stats)
            print(f"  Mean ODR `res_var` (Output Error Scale Factor): {mean_res_var:.4f}")
            if mean_res_var < 0.9:
                print("  Interpretation: `res_var` is significantly < 1, suggesting the input standard deviations (sx, sy) to ODR are OVERESTIMATED, leading to wider CIs.")
            elif mean_res_var > 1.1:
                print("  Interpretation: `res_var` is significantly > 1, suggesting the input standard deviations (sx, sy) to ODR are UNDERESTIMATED, leading to narrower CIs.")
            else:
                print("  Interpretation: `res_var` is close to 1, suggesting the input standard deviations to ODR are reasonable estimates of the true error.")
        elif regression_method == 'OLS':
            mean_sigma_squared_hat = np.mean(aux_regression_stats)
    
            # Recalculate expected_std_dev_f using the last pristine data
            if 'f_pristine' in results and results['f_pristine'] is not None:
                max_abs_f_pristine = np.max(np.abs(results['f_pristine']))
                expected_std_dev_f = noise_config['level_f'] * max_abs_f_pristine
                expected_variance_f_noise = expected_std_dev_f**2
                
                print(f"  Mean OLS `sigma_squared_hat` (Estimated Residual Variance): {mean_sigma_squared_hat:.4f}")
                print(f"  Expected Variance of Force Noise (from config): {expected_variance_f_noise:.4f}")
                if mean_sigma_squared_hat > 1.2 * expected_variance_f_noise:
                     print("  Interpretation: Estimated residual variance is significantly HIGHER than expected, suggesting input noise level might be underestimated or model fit issues.")
                elif mean_sigma_squared_hat < 0.8 * expected_variance_f_noise:
                     print("  Interpretation: Estimated residual variance is significantly LOWER than expected, suggesting input noise level might be overestimated, leading to wider CIs.")
                else:
                     print("  Interpretation: Estimated residual variance is close to expected noise variance, which is good.")
            else:
                print(f"  Mean OLS `sigma_squared_hat` (Estimated Residual Variance): {mean_sigma_squared_hat:.4f}")
                print("  Cannot compare to expected force noise variance as pristine force data is not available.")
    
    
        print("\n--- Empirical Std Dev vs. Mean Reported SE ---")
        print("  If Mean Reported SE is significantly larger than Empirical Std Dev, CIs are likely too wide (over-coverage).")
        print(f"  Parameter m: Empirical Std Dev={np.std(m_hats):.4f}, Mean Reported SE={np.mean(SE_m_hats):.4f}")
        print(f"  Parameter c: Empirical Std Dev={np.std(c_hats):.4f}, Mean Reported SE={np.mean(SE_c_hats):.4f}")
        print(f"  Parameter k: Empirical Std Dev={np.std(k_hats):.4f}, Mean Reported SE={np.mean(SE_k_hats):.4f}")
        
        # Get the total error
        true_error = results['last_true_error']
        measured_error = results['last_measured_error']
        bias = results['bias']

        # --- Analysis of Total Error ---

        if regression_method != 'KF':
            # Get the total error
            true_error = results.get('last_true_error')
            measured_error = results.get('last_measured_error')
            
            # Check if error terms exist and are not None before proceeding
            if true_error is not None and measured_error is not None:
                bias = true_error - measured_error

                # --- Analysis of Total Error ---

                # Plot Total Error vs. Time
                plt.figure(figsize=(12, 6))
                plt.plot(results['time_points'], true_error, label='True Error', color='blue', linewidth=2)
                plt.plot(results['time_points'], measured_error, label='Measured Error', color='red', linewidth=2)
                plt.plot(results['time_points'], bias, label='Bias', color='green', linewidth=2)
                plt.title('Total Error Over Time')
                plt.xlabel('Time (s)')
                plt.ylabel('Total Error (N)')
                plt.grid(True)
                plt.legend()
                plt.show()

                # Check for Time Dependence (e.g., using Pearson correlation coefficient)
                valid_indices = ~np.isnan(true_error)
                if np.sum(valid_indices) > 0:
                    correlation_coefficient, p_value_corr = stats.pearsonr(results['time_points'][valid_indices], true_error[valid_indices])
                    print(f"\n--- Total Error Time Dependence Analysis ---")
                    print(f"Pearson Correlation Coefficient (Total Error vs. Time): {correlation_coefficient:.4f}")
                    print(f"P-value for correlation: {p_value_corr:.4f}")
                    if p_value_corr < 0.05:
                        print("Conclusion: There is a statistically significant linear relationship between total error and time (p < 0.05).")
                    else:
                        print("Conclusion: No statistically significant linear relationship between total error and time (p >= 0.05).")
                else:
                    print("\n--- Total Error Time Dependence Analysis ---")
                    print("Cannot calculate correlation: Total error contains only NaN values.")


                # Plot Histogram of Total Error
                plt.figure(figsize=(8, 6))
                plt.subplot(1, 2, 1)
                plt.hist(true_error[valid_indices], bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Histogram of True Error')
                plt.xlabel('True Error (N)')
                plt.ylabel('Density')
                plt.grid(True)
        
                plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
                plt.hist(measured_error[valid_indices], bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
                plt.title('Histogram of Measured Error')
                plt.xlabel('Measured Error (N)')
                plt.ylabel('Density')
                plt.grid(True)

                plt.tight_layout()
                plt.show()

                # Q-Q Plot for Normality Check
                plt.figure(figsize=(8, 6))
                stats.probplot(true_error[valid_indices], dist="norm", plot=plt)
                plt.title('Q-Q Plot of Total Error Against Normal Distribution')
                plt.xlabel('Theoretical Quantiles')
                plt.ylabel('Ordered Values (Total Error)')
                plt.grid(True)
                plt.show()

                # Shapiro-Wilk Test for Normality
                if np.sum(valid_indices) > 3: # Shapiro-Wilk test requires at least 4 data points
                    shapiro_statistic, p_value_shapiro = stats.shapiro(true_error[valid_indices])
                    print(f"\n--- Total Error Normality Test (Shapiro-Wilk) ---")
                    print(f"Shapiro-Wilk Statistic: {shapiro_statistic:.4f}")
                    print(f"P-value: {p_value_shapiro:.4f}")
                    if p_value_shapiro < 0.05:
                        print("Conclusion: Total error is likely NOT normally distributed (p < 0.05).")
                    else:
                        print("Conclusion: Total error appears to be normally distributed (p >= 0.05).")
                else:
                    print("\n--- Total Error Normality Test (Shapiro-Wilk) ---")
                    print("Not enough data points (need at least 4) to perform Shapiro-Wilk test on total error.")

def plot_time_series(true_vals, noisy_vals, filtered_vals, pred_vals, time_points, title, ylabel, var_name, msd_system_params):
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, true_vals, label=f'True {var_name}', color='blue', linewidth=2)
    plt.plot(time_points, noisy_vals, 'o', markersize=2, alpha=0.6, label=f'{var_name} Estimated from Noisy x(t)', color='red')
    if filtered_vals is not None and not np.all(np.isnan(filtered_vals)):
        plt.plot(time_points, filtered_vals, label=f'Filtered {var_name}', color='black', linestyle='--', linewidth=2)
    if pred_vals is not None and not np.all(np.isnan(pred_vals)):
        plt.plot(time_points, pred_vals, label=f'Predicted {var_name} After Regression', color='green', linestyle='--', linewidth=2)
    plt.title(f'{title}\nTrue m={msd_system_params["m"]:.2f}, True c={msd_system_params["c"]:.2f}, k={msd_system_params["k"]:.2f}, x0={msd_system_params["x0"]:.2f}, v0={msd_system_params["v0"]:.2f}')
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_histograms(c_hats, k_hats, m_hats, true_c, true_k, true_m):
    plt.figure(figsize=(18, 5)) # Wider figure to accommodate three subplots
    
    plt.subplot(1, 3, 1) # 1 row, 3 columns, 1st plot
    plt.hist(m_hats, bins=30, density=True, alpha=0.6, color='lightgreen', edgecolor='black')
    plt.axvline(true_m, color='purple', linestyle='dashed', linewidth=2, label=f'True m={true_m:.2f}')
    plt.title('Distribution of Estimated Mass (m)')
    plt.xlabel('Estimated m')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2) # 1 row, 3 columns, 2nd plot
    plt.hist(c_hats, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    plt.axvline(true_c, color='red', linestyle='dashed', linewidth=2, label=f'True c={true_c:.2f}')
    plt.title('Distribution of Estimated Damping (c)')
    plt.xlabel('Estimated c')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3) # 1 row, 3 columns, 3rd plot
    plt.hist(k_hats, bins=30, density=True, alpha=0.6, color='salmon', edgecolor='black')
    plt.axvline(true_k, color='blue', linestyle='dashed', linewidth=2, label=f'True k={true_k:.2f}')
    plt.title('Distribution of Estimated Stiffness (k)')
    plt.xlabel('Estimated k')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()