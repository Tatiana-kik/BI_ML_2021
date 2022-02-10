import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# from knn import KNNClassifier
from metrics import binary_classification_metrics, multiclass_accuracy, accuracy_per_class
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
switch = '1'
binary_criteria = (y_train == '1') | (y_train == '0')
if switch == '1':
    binary_train_y = y_train[binary_criteria].astype(int)
    binary_train_X = x_train[binary_criteria]
else:
    binary_train_y = y_train
    binary_train_X = x_train
binary_criteria_test = (y_test == '1') | (y_test == '0')
if switch == '1':
    binary_test_y = y_test[binary_criteria_test].astype(int)
    binary_test_X = x_test[binary_criteria_test]
else:
    binary_test_y = y_test
    binary_test_X = x_test
import knn
# knn_classifier = KNNClassifier(k=3)
# knn_classifier.fit(binary_train_X, binary_train_y)
# # TODO: compute_distances_two_loops
# dists = knn_classifier.compute_distances_two_loops(binary_test_X)
# assert np.isclose(dists[0, 100], np.sum(np.abs(binary_test_X[0] - binary_train_X[100])))
# # TODO: compute_distances_one_loops
# dists = knn_classifier.compute_distances_one_loops(binary_test_X)
# assert np.isclose(dists[0, 100], np.sum(np.abs(binary_test_X[0] - binary_train_X[100])))
# # TODO: compute_distances_no_loops
# dists = knn_classifier.compute_distances_no_loops(binary_test_X)
# assert np.isclose(dists[0, 100], np.sum(np.abs(binary_test_X[0] - binary_train_X[100])))
# # TODO: predict_labels_binary in knn.py
# prediction = knn_classifier.predict(binary_test_X)
# print(binary_classification_metrics(prediction, binary_test_y))
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# print(accuracy_score(binary_test_y, prediction, normalize=False))
# if switch == '1':
#     average_parameter = 'binary'
# else:
#     average_parameter = 'micro'
# print(precision_score(binary_test_y, prediction, average = average_parameter))
# print(recall_score(binary_test_y, prediction, average = average_parameter))
# print(f1_score(binary_test_y, prediction, average = average_parameter))

knn_classifier = knn.KNNClassifier(k=5)
knn_classifier.fit(x_train, y_train)
prediction_multiclass = knn_classifier.predict(x_test)
acc_multiclass = accuracy_per_class(prediction_multiclass, y_test)
# plt.plot(acc_binary)
print(acc_multiclass)


