 # -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:13:08 2025

@author: Cormac Molyneaux
"""
#optimiser.py
import numpy as np
#from msd_system import MSDSystem
#from data_generator import generate_pristine_data, add_noise_to_data
#from regression_models import OLSModel, ODRModel
#from analysis import run_monte_carlo_sweep
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import KFold
from signal_processing import apply_butterworth_filter, differentiate_data_numerically
from skopt.space import Real, Integer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

# Define Low pass filter search space
lp_search_space = [
    Real(1.0, 25.0, name='cutoff_frequency', prior='log-uniform'),
    Integer(3, 11, name='order'),
    Real(1e-5, 1e5, name='lambda_val', prior='log-uniform')
]

class SplineWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, k=3, s=0.5):
        self.k = k
        self.s = s
        self.spline_ = None # Use trailing underscore for fitted attributes

    def fit(self, X, y):
        # Sort data, as UnivariateSpline requires it
        sorted_indices = np.argsort(X.ravel())
        X_sorted, y_sorted = X[sorted_indices], y[sorted_indices]
        
        # 'extrapolate' prevents errors when predicting on validation fold edges
        self.spline_ = UnivariateSpline(X_sorted, y_sorted, k=self.k, s=self.s, ext='extrapolate')
        return self

    def predict(self, X):
        return self.spline_(X)

def optimise_spline_hyperparameters(x, time_points):
    """
    Finds the optimal spline degree (k) and smoothing factor (s) using
    Grid Search with Cross-Validation, which is a statistical best practice.
    """
    # 1. Define the hyperparameter grid
    # A logarithmic space is often better for smoothing parameters like 's'
    param_grid = {
        'k': [3, 4, 5], # Cubic, quartic, quintic splines are common
        's': np.logspace(-3, 2, 50) # Search 's' from 0.001 to 100
        #'s': np.linspace(0.001, 5, 50)
    }
    
    # 2. Set up the model and cross-validation
    spline_model = SplineWrapper()
    
    # Use 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Perform Grid Search
    # GridSearchCV automatically handles the cross-validation loop and error scoring
    # 'neg_mean_squared_error' is used because GridSearchCV maximizes a score
    grid_search = GridSearchCV(
        estimator=spline_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        #n_jobs=-1
    )
    
    # The time_points must be reshaped for scikit-learn estimators
    grid_search.fit(time_points.reshape(-1, 1), x)
    
    # 4. Return the best parameters found
    #print(f"Best parameters found: {grid_search.best_params_}")
    #print(f"Best cross-validation score (negative MSE): {grid_search.best_score_:.4f}")
    
    best_k = grid_search.best_params_['k']
    best_s = grid_search.best_params_['s']
    
    return best_k, best_s

def low_pass_optimiser(x, time_points, dt, sampling_frequency):
    # --- 1. Define Hyperparameter grid ---
    # Important: cutoff_frequency must be < Nyquist frequency (sampling_frequency / 2)
    cutoff_frequencies = [1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0]
    orders = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    lambda_values = np.logspace(-4, 4, 9) # e.g., [0.0001, 0.001, ..., 1000, 10000]

    # --- 2. Set up K-Fold Cross-Validation ---
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    # --- 3. Search Grid ---
    best_combined_cost = float('inf')
    best_cutoff_frequency = None
    best_order = None
    best_lambda = None

    for lambda_val in lambda_values:
        for cutoff_frequency in cutoff_frequencies:
            for order in orders:
                fold_costs = []

                if cutoff_frequency >= (0.5 * sampling_frequency):
                    continue

                for train_index, val_index in kf.split(time_points):
                    x_noisy_val = x[val_index]

                    try:
                        x_filtered_val = apply_butterworth_filter(x_noisy_val, cutoff_frequency, sampling_frequency, order)
                        _, a_filtered_val = differentiate_data_numerically(x_filtered_val, dt)

                        MSE = np.mean((x_filtered_val - x_noisy_val)**2)

                        if len(a_filtered_val) > 1:
                            penalty_roughness = np.sum(np.diff(a_filtered_val)**2) / len(a_filtered_val)
                        else:
                            penalty_roughness = 0.0

                        # MODIFIED: Use the lambda_val from the current loop iteration
                        current_cost = MSE + (lambda_val * penalty_roughness)
                        fold_costs.append(current_cost)

                    except Exception as e:
                        # You can add a print statement here for debugging if needed
                        # print(f"Error for params ({lambda_val=}, {cutoff_frequency=}, {order=}): {e}")
                        fold_costs.append(float('inf'))
                        continue

                # Calculate average cost for the hyperparameter combination
                if len(fold_costs) > 0 and not all(c == float('inf') for c in fold_costs):
                    avg_fold_cost = np.mean([c for c in fold_costs if c != float('inf')])
                else:
                    avg_fold_cost = float('inf')

                results[(lambda_val, cutoff_frequency, order)] = avg_fold_cost

                # Update best parameters if current combination is better
                if avg_fold_cost < best_combined_cost:
                    best_combined_cost = avg_fold_cost
                    best_cutoff_frequency = cutoff_frequency
                    best_order = order
                    best_lambda = lambda_val 
                
    return best_cutoff_frequency, best_order