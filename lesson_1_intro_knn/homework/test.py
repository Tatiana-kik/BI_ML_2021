import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from knn import KNNClassifier
from metrics import binary_classification_metrics1, multiclass_accuracy1, accuracy_per_class, r_squared1
# SEED = 666
# random.seed(SEED)
# np.random.seed(SEED)
#X, y = fetch_openml(name="Fashion-MNIST", return_X_y=True, as_frame=False)
import pickle
X = pickle.load(open("X.txt", "rb"))
y = pickle.load(open("y.txt", "rb"))
idx_to_stay = np.random.choice(np.arange(X.shape[0]), replace=False, size=3000)
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
knn_classifier = KNNClassifier(k=3)
knn_classifier.fit(binary_train_X, binary_train_y)
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
prediction = knn_classifier.predict(binary_test_X)
print(accuracy_score(prediction, binary_test_y), recall_score(prediction, binary_test_y), precision_score(prediction, binary_test_y), f1_score(prediction, binary_test_y))
print(binary_classification_metrics1(prediction, binary_test_y))
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#print(accuracy_score(binary_test_y, prediction, normalize=False))
prediction_m = knn_classifier.predict(x_test)
print(accuracy_score(prediction_m, y_test), recall_score(prediction_m, y_test), precision_score(prediction_m, y_test), f1_score(prediction_m, y_test))
print(binary_classification_metrics1(prediction_m, y_test))
# if switch == '1':
#     average_parameter = 'binary'
# else:
#     average_parameter = 'micro'
# print(precision_score(binary_test_y, prediction, average = average_parameter))
# print(recall_score(binary_test_y, prediction, average = average_parameter))
# print(f1_score(binary_test_y, prediction, average = average_parameter))

# knn_classifier = knn.KNNClassifier(k=5)
# knn_classifier.fit(x_train, y_train)
# prediction_multiclass = knn_classifier.predict(x_test)
# acc_multiclass = accuracy_per_class(prediction_multiclass, y_test)
# # plt.plot(acc_binary)
# print(acc_multiclass)
# from sklearn.datasets import load_diabetes
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsRegressor
# X, y = load_diabetes(as_frame=True, return_X_y=True)
# from sklearn.pipeline import Pipeline
# x_train, x_test, y_train, y_test = train_test_split(X, y)
# regressorKN = KNeighborsRegressor(n_neighbors=5)
# pipelineRG = Pipeline(steps = [('regression', regressorKN)])
# pipelineRG.fit(x_train, y_train)
# print(r2_score(y_test, pipelineRG.predict(x_test)))
# print(pipelineRG.predict(x_test))
# print(r_squared1(pipelineRG.predict(x_test), y_test))
