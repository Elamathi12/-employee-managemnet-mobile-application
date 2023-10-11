import numpy as np

# Values in the dataset range from 1 to 1000
values = np.arange(1, 1001)

# Calculate mean and standard deviation
mean_value = np.mean(values)
std_value = np.std(values)

print("Mean:", mean_value)
print("Standard Deviation:", std_value)