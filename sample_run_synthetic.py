

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from KMeansClustering import KMeansClustering


data = make_blobs(n_samples=300,
                            centers=3,
                            cluster_std=0.90,
                            random_state=42)

X_data = data[0]  # Extracting the features (data points)
y_data = data[1]  # Extracting the labels (not used in clustering)

K_to_find = 3  
custom_kmeans_model = KMeansClustering(k=K_to_find, max_iters=100, random_state=42)

print("Fitting Custom K-Means model...")
custom_kmeans_model.fit(X_data)
print("Fitting complete.")

final_custom_labels = custom_kmeans_model.labels
final_custom_centroids = custom_kmeans_model.centroids


plt.figure(figsize=(8, 6))



plt.scatter(X_data[:, 0], X_data[:, 1], c=final_custom_labels, s=40, cmap='viridis', alpha=0.8, label='Data Points (Custom K-Means)')
 
plt.scatter(X_data[:, 0], X_data[:, 1], s=40, alpha=0.8, label='Data Points (Raw)')
if final_custom_centroids is not None:
    plt.scatter(final_custom_centroids[:, 0], final_custom_centroids[:, 1], c='red', s=250, marker='X', edgecolor='black', label='Centroids (Custom K-Means)')

plt.title(f'Custom K-Means Clustering (K={K_to_find})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

