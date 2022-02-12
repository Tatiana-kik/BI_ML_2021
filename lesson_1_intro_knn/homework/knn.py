import numpy as np
import pandas as pd
import scipy.stats


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, n_loops=1):
        """
        Uses the KNN model to predict classes for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        # if len(np.unique(self.train_y)) == 2:
        #     return self.predict_labels_binary(distances)
        # else:
        #     return self.predict_labels_multiclass(distances)

        return self.predict_labels_multiclass(distances)

    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        matrix_distance = np.zeros(shape=(len(X), len(self.train_X)))
        for u in range(0, len(X)):
            for v in range(0, len(self.train_X)):
                matrix_distance[u][v] = np.sum(np.abs(X[u, :] - self.train_X[v, :]))
        return matrix_distance

    def compute_distances_one_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        matrix_distance = np.zeros((num_test, num_train))
        for i in range(num_test):
            matrix_distance[i, :] = np.sum(np.abs(self.train_X - X[i, :]), axis=1)
        return matrix_distance

    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        matrix_distance = np.abs(X[:, None] - self.train_X).sum(-1)
        return matrix_distance

    # def predict_labels_binary(self, distances):
    #     """
    #     Returns model predictions for binary classification case
    #
    #     Arguments:
    #     distances, np array (num_test_samples, num_train_samples) - array
    #        with distances between each test and each train sample
    #     Returns:
    #     pred, np array of bool (num_test_samples) - binary predictions
    #        for every test sample
    #     """
    #
    #     n_train = distances.shape[1]
    #     n_test = distances.shape[0]
    #     prediction = np.zeros(n_test)
    #
    #     """
    #     YOUR CODE IS HERE
    #     """
    #     pass

    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Argument:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """
        n_test = distances.shape[0]
        prediction = ['']*n_test
        for t in range(0, n_test):
            neighbours = ['']*self.k
            temp_dist = distances[t].copy()
            for i in range(0, self.k):
                min_index = np.nanargmin(temp_dist)
                neighbours[i] = self.train_y[min_index]
                temp_dist[min_index] = np.nan
            prediction[t] = scipy.stats.mode(neighbours).mode[0]
        return prediction
