# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:04:44 2025

@author: Cormac Molyneaux
"""

# run_big_sim.py
import subprocess
import itertools

def run_monte_carlo_simulations():
    # Define the specific m, c, k combinations for each damping case
    damping_cases = {
        'ud': {'m': 1.0, 'c': 2.0, 'k': 9.0}, # Under-damped
        'cd': {'m': 1.0, 'c': 6.0, 'k': 9.0}, # Critically-damped
        'od': {'m': 2.0, 'c': 6.0, 'k': 2.0}  # Over-damped
    }

    # Extract the damping case names to iterate through
    damping_case_names = ['ud', 'cd', 'od']
    # Example for file_name - you might want to generate unique names
    #file_name_prefix = 'simulation_results'
    filter_types = ['gaussian_proccess']
    regression_methods = ['OLS', 'ODR']

    # Generate all combinations of parameters
    # The order here should match the order you expect for clarity in the loop
    param_combinations = list(itertools.product(
        damping_case_names,
        filter_types,
        regression_methods
    ))

    print(f"Starting {len(param_combinations)} Monte Carlo simulations...")

    for i, (damping_case_name, filter_type, regression_method) in enumerate(param_combinations):
        # Get the specific m, c, k values for the current damping case
        current_params = damping_cases[damping_case_name]
        m = current_params['m']
        c = current_params['c']
        k = current_params['k']
        
        current_file_name = f"{filter_type}_{regression_method}_{damping_case_name}.pkl"

        print(f"\n--- Running Simulation {i+1}/{len(param_combinations)} ---")
        print(f"Parameters: m={m}, c={c}, k={k}, Filter={filter_type}, Regression={regression_method}, File={current_file_name}")

        # Construct the command to run main.py with specific arguments
        command = [
            'python', 'main.py',
            '--m', str(m),
            '--c', str(c),
            '--k', str(k),
            '--file_name', current_file_name,
            '--filter_type', filter_type,
            '--regression_method', regression_method
        ]

        try:
            # Execute main.py as a subprocess
            # capture_output=True and text=True are good for debugging if main.py prints
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("main.py stdout:\n", result.stdout)
            if result.stderr:
                print("main.py stderr:\n", result.stderr)
            print(f"Simulation {i+1} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Simulation {i+1} failed with error:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"Error: 'python' command not found. Ensure Python is in your system's PATH.")
            break
        except Exception as e:
            print(f"An unexpected error occurred during simulation {i+1}: {e}")
            break

    print("\nAll Monte Carlo simulations finished.")

if __name__ == "__main__":
    run_monte_carlo_simulations()