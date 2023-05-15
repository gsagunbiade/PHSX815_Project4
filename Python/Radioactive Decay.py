# -*- coding: utf-8 -*-
"""
Created on Sat May 13 19:50:28 2023

@author: Gbenga Agunbiade
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# Simulating radioactive decay data
np.random.seed(1)
decay_constant = 0.2  # Decay constant (rate) of the radioactive substance
time_interval = 1  # Time interval for data collection
measurement_time = 250  # Total time of data collection
num_measurements = int(measurement_time / time_interval)

# Generate decay counts based on Poisson fluctuations around the expected value
expected_counts = decay_constant * time_interval * np.arange(0, num_measurements)
decay_counts = np.random.poisson(expected_counts)

# Analyzing the decay counts
mean_counts = np.mean(decay_counts)
std_dev_counts = np.std(decay_counts)

# Plotting the decay counts over time
time = np.arange(0, num_measurements) * time_interval
plt.figure(figsize=(8, 6))
plt.plot(time, decay_counts, 'green', label='Observed Counts')
plt.plot(time, expected_counts, 'red', label='Expected Counts')
plt.xlabel('Time')
plt.ylabel('Decay Counts')
plt.legend()
plt.title('Radioactive Decay')
plt.show()

# Plotting the histogram and fitted Gaussian curve
plt.figure(figsize=(6, 4))
plt.hist(decay_counts, bins=10, density=True, alpha=0.7, label='Observed Data')
x = np.linspace(np.min(decay_counts), np.max(decay_counts), 100)
fit_mu, fit_sigma = norm.fit(decay_counts)
plt.plot(x, norm.pdf(x, fit_mu, fit_sigma), 'blue', label='Gaussian Fit')
plt.xlabel('Decay Counts')
plt.ylabel('Density')
plt.legend()
plt.title('Histogram of Decay Counts')
plt.show()

# Output statistical properties
print("Mean Counts:", mean_counts)
print("Standard Deviation of Counts:", std_dev_counts)
