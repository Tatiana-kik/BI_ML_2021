import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from knn import KNNClassifier
from metrics import binary_classification_metrics, multiclass_accuracy
SEED = 666
random.seed(SEED)
np.random.seed(SEED)
#X, y = fetch_openml(name="Fashion-MNIST", return_X_y=True, as_frame=False)
import pickle
X = pickle.load(open("X.txt", "rb"))
y = pickle.load(open("y.txt", "rb"))
idx_to_stay = np.random.choice(np.arange(X.shape[0]), replace=False, size=1000)
X = X[idx_to_stay]
y = y[idx_to_stay]
x_train, x_test, y_train, y_test = train_test_split(X, y)
criteria = (y_train == '1') | (y_train == '0')
binary_train_y = y_train#[criteria]
binary_train_X = x_train#[criteria]
criteria_test = (y_test == '1') | (y_test == '0')
binary_test_y = y_test#[criteria_test]
binary_test_X = x_test#[criteria_test]
import knn
knn_classifier = KNNClassifier(k=1)
knn_classifier.fit(binary_train_X, binary_train_y)
# TODO: compute_distances_two_loops
dists = knn_classifier.compute_distances_two_loops(binary_test_X)
assert np.isclose(dists[0, 100], np.sum(np.abs(binary_test_X[0] - binary_train_X[100])))
# TODO: compute_distances_one_loop
dists = knn_classifier.compute_distances_one_loop(binary_test_X)
assert np.isclose(dists[0, 100], np.sum(np.abs(binary_test_X[0] - binary_train_X[100])))
# TODO: compute_distances_no_loops
dists = knn_classifier.compute_distances_no_loops(binary_test_X)
assert np.isclose(dists[0, 100], np.sum(np.abs(binary_test_X[0] - binary_train_X[100])))


