# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:34:00 2025

@author: Cormac Molyneaux
"""
# msd_system.py
import numpy as np

class MSDSystem:
    def __init__(self, m=1.0, c=2.0, k=9.0, 
                 x0=1.0, v0=-0.2, 
                 force_amplitude=0.0, force_frequency=0.0):
        self.m = m
        self.c = c
        self.k = k
        self.x0 = x0
        self.v0 = v0
        self.force_amplitude = force_amplitude
        self.force_frequency = force_frequency

        if self.m == 0:
            raise ValueError("Mass (m) cannot be zero.")
        if self.k < 0:
             print("Warning: Negative spring constant, system might be unstable.")

        # Pre-calculate natural frequency and damping ratio if m=1
        self.omega_n = np.sqrt(self.k / self.m)
        self.zeta = self.c / (2 * self.m * self.omega_n) if self.omega_n > 0 else np.inf

    def get_continuous_dynamics(self, state, t, msd_system_instance):
        """
        Defines the continuous-time dynamics of the MSD system.
        Used by ODE solvers (e.g., scipy.integrate.odeint).
        State is [position, velocity]. Returns [d_position/dt, d_velocity/dt].
        """
        x, v = state

        # Forcing term
        F_t = self.force_amplitude * np.cos(self.force_frequency * t)

        dxdt = v
        dvdt = (F_t - self.c * v - self.k * x) / self.m

        return [dxdt, dvdt]

    def get_acceleration(self, x, v, t):
        """Calculates instantaneous acceleration based on system parameters."""
        F_t = self.force_amplitude * np.cos(self.force_frequency * t)
        return (F_t - self.c * v - self.k * x) / self.m
