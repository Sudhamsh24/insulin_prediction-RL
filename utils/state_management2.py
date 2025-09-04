# utils/state_management2.py

import collections
import numpy as np
from scipy.stats import gamma

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    """
    Pharmacokinetics/Pharmacodynamics insulin absorption curve
    """
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

class StateRewardManager:
    def __init__(self, state_dim):
        self.glucose_history = collections.deque(maxlen=12) 
        self.insulin_history = collections.deque(maxlen=160)
        self.reset()
        _, self.F_k = get_pkpd_discount_factors(
            t_peak=55, t_end=480, n_steps=160
        )
        self.running_state_mean = np.zeros(state_dim)
        self.running_state_std = np.ones(state_dim)
        self.n_observations = 0

    def update_normalization_stats(self, state):
        self.n_observations += 1
        old_mean = self.running_state_mean.copy()
        self.running_state_mean += (state - self.running_state_mean) / self.n_observations
        self.running_state_std += (state - old_mean) * (state - self.running_state_mean)

    def get_normalized_state(self, state):
        self.update_normalization_stats(state)
        std = np.sqrt(self.running_state_std / (self.n_observations if self.n_observations > 1 else 1))
        return (state - self.running_state_mean) / (std + 1e-8)

    def calculate_iob(self):
        """
        Insulin On Board from history and PK/PD curve
        """
        return np.sum(np.array(list(self.insulin_history)[::-1]) * (1 - self.F_k))

    def get_full_state(self, observation, upcoming_carbs=0):
        """
        observation: glucose level (mg/dL)
        """
        self.glucose_history.append(observation)
        if len(self.glucose_history) == self.glucose_history.maxlen:
            rate = (self.glucose_history[-1] - self.glucose_history[0]) / (self.glucose_history.maxlen * 5.0 / 60.0)
        else:
            rate = 0.0
        iob = self.calculate_iob()
        return np.array([observation, rate, iob, upcoming_carbs])

    def get_reward(self, state):
        g, rate, iob, _ = state
    
        # 1) Time-in-range bonus
        if 80 <= g <= 140:
            tir_bonus = 2.0
        elif 70 <= g < 80 or 140 < g <= 180:
            tir_bonus = 1.0
        else:
            tir_bonus = 0.0
    
        # 2) Proximity to 110 with broader sigma
        target = 110.0
        sigma  = 30.0
        proximity = 5.0 * np.exp(-0.5 * ((g - target) / sigma) ** 2)
    
        reward = tir_bonus + proximity
    
        # ! REVISED: Stronger and more immediate hypo penalty
        if g < 80: # Start penalizing as soon as glucose drops below 80
            reward -= 100.0 * (1 + (80 - g) / 10.0)
        if g < 70: # Very strong additional penalty for clinical hypoglycemia
            reward -= 500.0 * (1 + (70 - g) / 5.0)

        # 4) Hyper penalty (no hard cap; quadratic after 180) - ! IMPROVED PENALTY
        if g > 180:
            over = g - 180.0
            if g <= 220:
                reward -= 5.0 * over
            else:
                reward -= 40.0 + 0.1 * (g - 220.0) ** 2
    
        # 5) Penalize excessive IOB and sharp swings
        reward -= 0.25 * (iob ** 2)
        reward -= 0.05 * (abs(rate) ** 1.5)
    
        # 6) ! NEW: Penalty for large insulin boluses to encourage smoother action
        if len(self.insulin_history) >= 2:
            bolus_size = self.insulin_history[-1]
            if bolus_size > 2.0: # Penalize boluses over 2 units
                reward -= 0.5 * (bolus_size - 2.0)
            du = abs(self.insulin_history[-1] - self.insulin_history[-2])
            reward -= 0.02 * du
    
        return reward

    def reset(self):
        self.glucose_history.clear()
        for _ in range(self.glucose_history.maxlen):
            self.glucose_history.append(140)
        self.insulin_history.clear()
        for _ in range(160):
            self.insulin_history.append(0)