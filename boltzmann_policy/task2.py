from task1 import frozen_lake
import numpy as np
from scipy import stats

successful_episodes = []

while len(successful_episodes) < 10:
    run = frozen_lake(False, False, 1000000)[0]
    if run[0] == True:
        successful_episodes.append(run[1])

# Calculate sample mean and standard deviation
sample_mean = np.mean(successful_episodes)
sample_std = np.std(successful_episodes, ddof=1)  # Use ddof=1 for sample standard deviation

# Set the desired level of confidence (e.g., 95% confidence interval)
confidence_level = 0.95

# Calculate the critical value from the t-distribution
degrees_of_freedom = len(successful_episodes) - 1
t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

# Calculate the margin of error
margin_of_error = t_critical * (sample_std / np.sqrt(len(successful_episodes)))

# Calculate the confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print("Sample Mean:", sample_mean)
print("Confidence Interval:", confidence_interval)