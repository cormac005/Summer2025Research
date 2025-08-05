# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:44:45 2025

@author: corma
"""
#Data_Analysis
import pickle
import os
import matplotlib.pyplot as plt
import glob

def load_pickle_file(filepath):
    """
    Loads data from a single .pkl file.

    Args:
        filepath (str): The path to the .pkl file.

    Returns:
        dict: The data loaded from the file, or None if an error occurs.
    """
    try:
        with open(filepath, 'rb') as f:
            # Use 'latin1' encoding for compatibility with older pickle files
            data = pickle.load(f, encoding='latin1')
        return data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def plot_time_series(data, base_name, output_dir):
    """
    Generates and saves a time series plot for x, v, a, and f.

    Args:
        data (dict): The data loaded from a .pkl file.
        base_name (str): The base name of the input file for titling.
        output_dir (str): The directory to save the plot in.
    """
    if 'time_points' not in data:
        print(f"Skipping time series plot for {base_name}: 'time_points' not found.")
        return

    time_points = data['time_points']
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Time Series Analysis for {base_name}', fontsize=16)

    # Plot 1: Displacement (x)
    axes[0].set_title('Displacement (x)')
    # Added zorder to ensure the 'True' line is on top of the 'Noisy' points
    # Also reduced markersize for the noisy points for better visibility
    axes[0].plot(time_points, data.get('x_pristine', []), 'k-', label='True', linewidth=2, zorder=2)
    axes[0].plot(time_points, data.get('last_x_noisy', []), 'r.', label='Noisy', alpha=0.6, markersize=2, zorder=1)
    if 'last_x_filtered' in data:
        axes[0].plot(time_points, data['last_x_filtered'], 'b--', label='Filtered', linewidth=2, zorder=3)
    axes[0].set_ylabel('Displacement')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Velocity (v)
    axes[1].set_title('Velocity (v)')
    # Added zorder for consistency
    axes[1].plot(time_points, data.get('v_pristine', []), 'k-', label='True', linewidth=2, zorder=2)
    axes[1].plot(time_points, data.get('last_v_noisy', []), 'b--', label='Estimated', alpha=0.6, zorder=1)
    axes[1].set_ylabel('Velocity')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Acceleration (a)
    axes[2].set_title('Acceleration (a)')
    # Added zorder for consistency
    axes[2].plot(time_points, data.get('a_pristine', []), 'k-', label='True', linewidth=2, zorder=2)
    axes[2].plot(time_points, data.get('last_a_noisy', []), 'b--', label='Estimated', alpha=0.6, zorder=1)
    axes[2].set_ylabel('Acceleration')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Force (f)
    axes[3].set_title('Force (f)')
    # Added zorder and reduced markersize
    axes[3].plot(time_points, data.get('f_pristine', []), 'k-', label='True', linewidth=2, zorder=2)
    axes[3].plot(time_points, data.get('last_f_noisy', []), 'r.', label='Noisy', alpha=0.6, markersize=2, zorder=1)
    axes[3].set_ylabel('Force')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()
    axes[3].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(output_dir, f'timeseries_{base_name}.png')
    plt.savefig(plot_filename)
    plt.close(fig) # Close the figure to free up memory
    print(f"  - Saved time series plot to {plot_filename}")

def plot_histograms(data, base_name, output_dir):
    """
    Generates and saves histograms for m_hats, c_hats, and k_hats.

    Args:
        data (dict): The data loaded from a .pkl file.
        base_name (str): The base name of the input file for titling.
        output_dir (str): The directory to save the plot in.
    """
    # Corrected the logic to determine true_values using a dictionary for better readability
    true_values_map = {
        'u': [1.0, 2.0, 9.0],
        'c': [1.0, 6.0, 9.0],
        'o': [2.0, 6.0, 2.0]
    }
    # Using .get() for safer access
    true_values = true_values_map.get(base_name[-2], [0.0, 0.0, 0.0])

    params = ['m_hats', 'c_hats', 'k_hats']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Parameter Estimation Histograms for {base_name}', fontsize=16)

    for i, param in enumerate(params):
        if param in data:
            true_value = true_values[i]
            ax = axes[i]
            ax.hist(data[param], bins=30, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of {param}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.axvline(true_value, color='g', linestyle='--', linewidth=2, label=f'True Value: {true_value:.1f}')
            # Added this line to display the legend with the true value label
            ax.legend()
        else:
            print(f"Warning: '{param}' not found in {base_name}. Skipping histogram.")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_filename = os.path.join(output_dir, f'histograms_{base_name}.png')
    plt.savefig(plot_filename)
    plt.close(fig) # Close the figure to free up memory
    print(f"  - Saved histogram plot to {plot_filename}")


def generate_individual_plots(directory='.', output_dir='plots'):
    """
    Finds all .pkl files, loads them, and generates time series and
    histogram plots for each file.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find all .pkl files in the specified directory
    file_paths = glob.glob(os.path.join(directory, '*.pkl'))
    
    if not file_paths:
        print("No .pkl files found in the current directory.")
        return

    # Process each file
    for path in file_paths:
        base_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\nProcessing file: {base_name}.pkl")

        data = load_pickle_file(path)
        if data:
            # Generate the time series plot for the current file
            plot_time_series(data, base_name, output_dir)
            
            # Generate the histogram plots for the current file
            plot_histograms(data, base_name, output_dir)

if __name__ == '__main__':
    # This will run the analysis on all .pkl files in the same
    # directory as the script and save the plots in a 'plots' subfolder.
    generate_individual_plots()
