import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

# Example data
x = np.array([0.95, 0.99, 1.02, 1, 1.01, 0.97, 1.99, 2.03, 1.98, 1.97, 2.01, 4.03, 4.02, 4.04, 3.97, 3.98])

# Step 1: Smooth the data
smoothed_x = gaussian_filter1d(x, sigma=0.2)

# Step 2: Detect steps using the derivative
derivative = np.diff(smoothed_x)
threshold = 0.2  # Adjust this threshold based on your data
steps = np.where(np.abs(derivative) > threshold)[0] + 1

# Step 3: Cluster the data points
kmeans = KMeans(n_clusters=len(steps) + 1)
x_reshaped = x.reshape(-1, 1)
kmeans.fit(x_reshaped)
labels = kmeans.labels_

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, label='Original Data', marker='o')
plt.plot(smoothed_x, label='Smoothed Data', linestyle='--')
plt.scatter(steps, smoothed_x[steps], color='red', label='Detected Steps')
for i in range(len(steps) + 1):
	plt.scatter(np.where(labels == i), x[labels == i], label=f'Bin {i+1}')
plt.legend()
plt.show()
