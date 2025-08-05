# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:34:16 2025

@author: Cormac Molyneaux
"""
# regression_models.py
import numpy as np
from scipy import odr 
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

# Define the ODR model function for the FORCED case (3 parameters: m, c, k)
def _msd_odr_model_forced(beta, a_v_x_data):
    m_val, c_val, k_val = beta
    a = a_v_x_data[0]
    v = a_v_x_data[1]
    x = a_v_x_data[2]
    return m_val * a + c_val * v + k_val * x

# Define the ODR model function for the UNFORCED case (2 parameters: c_over_m, k_over_m)
def _msd_odr_model_unforced(beta, v_and_x):
    c_over_m, k_over_m = beta
    v = v_and_x[0]
    x = v_and_x[1]
    return c_over_m * v + k_over_m * x

class OLSModel:
    def fit(self, x, v, a, f):
        
        X_matrix = np.column_stack((a, v, x))
        Y_vector = f

        Xt = X_matrix.T
        XtX = np.dot(Xt, X_matrix)
        XtY = np.dot(Xt, Y_vector)

        if np.linalg.det(XtX) == 0:
            print("Warning: X^T * X matrix is singular.")
            return [np.nan]*7

        B = np.linalg.solve(XtX, XtY)
        m_hat, c_hat, k_hat = B[0], B[1], B[2]
        
        # Calculate residuals
        residuals = Y_vector - np.dot(X_matrix, B)
        
        # Estimate variance of the error term (sigma_squared_hat)
        n = len(Y_vector)
        n_params = len(B)
        df_residuals = n - n_params
        
        if df_residuals > 0:
            sigma_squared_hat = np.sum(residuals**2) / df_residuals
        else:
            sigma_squared_hat = np.nan

        # Calculate covariance matrix of coefficients
        cov_B = sigma_squared_hat * np.linalg.inv(XtX)

        SE_m_hat = np.sqrt(cov_B[0, 0])
        SE_c_hat = np.sqrt(cov_B[1, 1])
        SE_k_hat = np.sqrt(cov_B[2, 2])
        
        # Determine if the system is unforced from the noisy force data 'f'
        # A small absolute threshold for general applicability, adjust as needed.
        force_threshold = 1e-3

        is_unforced = np.max(np.abs(f)) < force_threshold

        if is_unforced:
            # Scale c_hat and k_hat by m_hat only when the system is unforced.
            # This is because when F=0, only ratios c/m and k/m are uniquely identifiable.
            # Assuming m=1 provides a unique solution for c and k.
            if m_hat != 0: # Avoid division by zero
                c_hat = c_hat / m_hat
                k_hat = k_hat / m_hat
                SE_c_hat = SE_c_hat / m_hat
                SE_k_hat = SE_k_hat / m_hat
            else:
                # Handle case where m_hat is zero (highly unlikely for well-behaved data)
                c_hat = np.nan
                k_hat = np.nan
                SE_c_hat = np.nan
                SE_k_hat = np.nan

        return m_hat, c_hat, k_hat, SE_m_hat, SE_c_hat, SE_k_hat, sigma_squared_hat, df_residuals

class ODRModel:
    def fit(self, x, v, a, f, std_dev_x=None, std_dev_v=None, std_dev_a=None, std_dev_f=None, beta0_guess=None):

        # Determine if the system is unforced from the noisy force data 'f'
        # A small absolute threshold for general applicability, adjust as needed.
        force_threshold = max(1e-6, 4.5*std_dev_f)

        is_unforced = np.max(np.abs(f)) < force_threshold

       # --- ODR Model Setup ---
        if is_unforced:
            # UNFORCED CASE: Fit -a = (c/m)*v + (k/m)*x
            # Parameters will be [c_over_m, k_over_m]. (We assume m = 1.0)
            # Dependent variable: -a
            # Independent variables: [v, x]
            dependent_data = -a
            independent_data = np.row_stack((v, x))
            linear_odr_model = odr.Model(_msd_odr_model_unforced)
            
            # Initial guess for [c/m, k/m]
            if beta0_guess is None or len(beta0_guess) != 2: # Adjust beta0_guess expectation
                beta0_guess_for_unforced = [np.random.uniform(0.1, 5.0), np.random.uniform(1.0, 20.0)]
            else:
                beta0_guess_for_unforced = beta0_guess[1:] # Assuming beta0_guess has m,c,k and we take c,k

            # Standard deviations for sx: [std_dev_v, std_dev_x]
            sx_for_unforced = None
            if std_dev_x is not None and std_dev_v is not None:
                sx_for_unforced = np.array([std_dev_v, std_dev_x])
            
            # Standard deviation for sy: std_dev_a (as -a is the dependent var)
            sy_for_unforced = std_dev_a # Noise on 'a'

            data = odr.RealData(independent_data, dependent_data, sx=sx_for_unforced, sy=sy_for_unforced)
            myodr = odr.ODR(data, linear_odr_model, beta0=beta0_guess_for_unforced)
            myoutput = myodr.run()

            c_over_m_hat, k_over_m_hat = myoutput.beta
            SE_c_over_m_hat, SE_k_over_m_hat = myoutput.sd_beta
            res_var = myoutput.res_var
            
            m_hat = 1.0 # Conceptually fixed for unforced case
            c_hat = c_over_m_hat * m_hat 
            k_hat = k_over_m_hat * m_hat
            
            SE_m_hat = np.nan
            SE_c_hat = SE_c_over_m_hat * m_hat 
            SE_k_hat = SE_k_over_m_hat * m_hat 
            
            n_params = 2 # We estimated 2 parameters (c/m, k/m)
            df = len(x) - n_params

        else:
            # FORCED CASE: Fit F = m*a + c*v + k*x
            # Parameters will be [m, c, k]
            
            dependent_data = f
            independent_data = np.row_stack((a, v, x)) # Order: a, v, x
            linear_odr_model = odr.Model(_msd_odr_model_forced) # Use the 3-param model

            if beta0_guess is None or len(beta0_guess) != 3: # Default guess for 3 params
                # Use np.random.uniform for generating float initial guesses
                initial_guesses = [np.random.uniform(0.5, 2.0), np.random.uniform(0.5, 5.0), np.random.uniform(5.0, 15.0)]
            else:
                initial_guesses = beta0_guess
                
            # Explicitly convert to a NumPy array to ensure correct shape and type
            beta0_odr = np.array(initial_guesses, dtype=float)

            sx_for_forced = None
            if std_dev_x is not None and std_dev_v is not None and std_dev_a is not None:
                sx_for_forced = np.array([std_dev_a, std_dev_v, std_dev_x]) # Order: std_dev_a, std_dev_v, std_dev_x
            
            sy_for_forced = np.full(std_dev_x.shape, std_dev_f)
             
            data = odr.RealData(independent_data, dependent_data, sx=sx_for_forced, sy=sy_for_forced)
            myodr = odr.ODR(data, linear_odr_model, beta0=beta0_odr)
            myoutput = myodr.run()

            m_hat, c_hat, k_hat = myoutput.beta
            SE_m_hat, SE_c_hat, SE_k_hat = myoutput.sd_beta
            res_var = myoutput.res_var
            
            n_params = 3 # We estimated 3 parameters (m, c, k)
            df = len(x) - n_params

        return m_hat, c_hat, k_hat, SE_m_hat, SE_c_hat, SE_k_hat, res_var, df