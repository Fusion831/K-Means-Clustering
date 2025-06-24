import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    """Implementation of k-Means Clustering algorithm."""
    
    def __init__(self, k=3, max_iters=100, random_state=None,init_method='random_points'):
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
        init_method : str, optional
            Method for initializing centroids. Default is 'random_points', which randomly selects k data points
        """
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None # Will store the final centroids
        self.labels = None    # Will store the cluster labels for each data point
        self.init_method = init_method
    
    
    def euclidean_distance(self, a, b):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt(np.sum((a - b) ** 2))
    
    def _initialize_centroids(self, X):
        """
        Initializes k centroids based on the specified method.

        Parameters:
        
        X : np.ndarray
            Data points, shape (n_samples, n_features).
        """
        if self.init_method == 'random_points':
            self._init_random_points(X)
        elif self.init_method == 'random_range':
            self._init_random_range(X)
        elif self.init_method == 'kmeans++':
            self._init_kmeans_plus_plus(X)
            
        
    
    
    def _init_random_range(self, X):
        """
        Initializes k centroids by randomly selecting k points within the range of the data.

        Parameters:
        
        X : np.ndarray
              Data points, shape (n_samples, n_features).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state) 

        n_features = X.shape[1]
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)

        self.centroids = np.random.uniform(min_vals, max_vals, (self.k, n_features))    
    
    
    
    
    def _init_random_points(self, X):
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
        
    def _init_kmeans_plus_plus(self, X):
        n_samples, _ = X.shape
        chosen_centroids_coords = []

        
        first_idx = np.random.choice(n_samples)
        chosen_centroids_coords.append(X[first_idx])

        # Stores squared distance of each point to its nearest already chosen centroid
        min_sq_distances_to_chosen = np.full(n_samples, np.inf)

        
        for num_already_chosen in range(1, self.k):
            if num_already_chosen >= n_samples and self.k > n_samples:
                
                chosen_centroids_coords.append(chosen_centroids_coords[-1])
                continue

            last_added_centroid = chosen_centroids_coords[-1]
            
            
            for j in range(n_samples):
                dist_sq = np.sum((X[j] - last_added_centroid)**2)
                min_sq_distances_to_chosen[j] = min(min_sq_distances_to_chosen[j], dist_sq)

            sum_sq_dist = np.sum(min_sq_distances_to_chosen)
            
            if sum_sq_dist == 0:
                
                #To pick a point not yet chosen as a centroid
                temp_chosen_arr = np.array(chosen_centroids_coords)
                available_indices = [
                    i for i in range(n_samples) 
                    if not np.any(np.all(X[i] == temp_chosen_arr, axis=1, keepdims=True).T & (X[i].shape == temp_chosen_arr.shape[1:]))
                ] 
                candidate_indices = []
                current_centroids_set = [tuple(c) for c in chosen_centroids_coords]
                for i in range(n_samples):
                    if tuple(X[i]) not in current_centroids_set:
                        candidate_indices.append(i)
                
                if not candidate_indices: 
                    next_idx = np.random.choice(n_samples) 
                else:
                    next_idx = np.random.choice(candidate_indices) 
            else:
                probabilities = min_sq_distances_to_chosen / sum_sq_dist
                next_idx = np.random.choice(n_samples, p=probabilities)
            
            chosen_centroids_coords.append(X[next_idx])

        self.centroids = np.array(chosen_centroids_coords)

        
    
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
    
    
    def _calculate_inertia(self, X, labels):
        """Calculates the Within-Cluster Sum of Squares (WCSS) or Inertia."""
        if self.centroids is None or labels is None:
            return np.inf # Or some other indicator that it can't be calculated

        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # Sum of squared distances from points to their assigned centroid
                inertia += np.sum((cluster_points - self.centroids[i])**2)
        return inertia
    
    
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
        
        self.inertia = self._calculate_inertia(X, self.labels)

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