import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    """Implementation of k-Means Clustering algorithm."""
    
    def __init__(self, k=3, max_iters=100, random_state=None):
        """
        k-Means Clustering Algorithm.

        Parameters:
        
        k : int
            The number of clusters to form.
        max_iters : int
            Maximum number of iterations of the k-means algorithm for a single run.
        random_state : int, optional
            Determines random number generation for centroid initialization. Use an int to make
            the randomness deterministic.
        """
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None # Will store the final centroids
        self.labels = None    # Will store the cluster labels for each data point
    
    
    def euclidean_distance(self, a, b):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt(np.sum((a - b) ** 2))
    
    
    def _initialize_centroids(self, X):
        """
        Initializes k centroids by randomly selecting k data points from X.

        Parameters:
        
        X : np.ndarray
            Data points, shape (n_samples, n_features).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state) 

        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        
    
    def _assign_clusters(self, X):
        """
        Assigns each data point to the nearest centroid.

        Parameters:
        
        X : np.ndarray
            Data points, shape (n_samples, n_features).
        self.centroids : np.ndarray
            Current centroids, shape (k, n_features). Assumed to be initialized.

        Returns:
        
        np.ndarray
            Array of cluster labels for each data point, shape (n_samples,).
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        if self.centroids is None:
            raise ValueError("Centroids have not been initialized. Call '_initialize_centroids' first.")

        for i in range(n_samples):
            data_point = X[i]
            
            distances = [self.euclidean_distance(data_point, centroid) for centroid in self.centroids]

            labels[i] = np.argmin(distances)

        return labels
    
    
    
    
    def _update_centroids(self, X, labels):
        """
        Updates the centroids based on the current cluster assignments.

        Parameters:
        
        X : np.ndarray
            Data points, shape (n_samples, n_features).
        labels : np.ndarray
            Current cluster labels for each data point, shape (n_samples,).
        Returns:
        
        np.ndarray
            Updated centroids, shape (k, n_features).
        """
        new_centroids = np.zeros((self.k, X.shape[1]))
        
        for k in range(self.k):
            points_in_cluster = X[labels == k]
            if len(points_in_cluster) > 0:
                new_centroids[k] = np.mean(points_in_cluster, axis=0)
            else:
                new_centroids[k] = X[np.random.choice(X.shape[0])]
        
        self.centroids = new_centroids
        return new_centroids
    
    
    def fit(self, X):
        """
        Fit the k-means model to the data.

        Parameters:
        
        X : np.ndarray
            Data points, shape (n_samples, n_features).
        """
        self._initialize_centroids(X)

        for _ in range(self.max_iters):
            
            self.labels = self._assign_clusters(X)

            
            new_centroids = self._update_centroids(X, self.labels)

            
            if np.all(new_centroids == self.centroids):
                break

        return self.labels, self.centroids
    
    
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
        
        X : np.ndarray
            Data points, shape (n_samples, n_features).

        Returns:
        
        np.ndarray
            Predicted cluster labels for each data point, shape (n_samples,).
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")

        return self._assign_clusters(X)