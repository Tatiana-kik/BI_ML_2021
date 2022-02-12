import numpy as np
import statistics


def binary_classification_metrics1(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    metrics = {}
    metrics['accuracy'] = multiclass_accuracy1(y_pred, y_true)
    return metrics


def multiclass_accuracy1(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    assert(len(y_pred) == len(y_true))
    number_of_classes = len(set(y_true))
    TP = [0] * number_of_classes
    FP = [0] * number_of_classes
    TN = [0] * number_of_classes
    FN = [0] * number_of_classes
    for cl in range(0, number_of_classes):
        for i in range(0, len(y_true)):
            TP[cl] += int((y_pred[i] == y_true[i]) and str(y_pred[i]) == str(cl))
            FP[cl] += int((y_pred[i] != y_true[i]) and str(y_pred[i]) == str(cl))
            TN[cl] += int((y_pred[i] == y_true[i]) and str(y_pred[i]) != str(cl))
            FN[cl] += int((y_pred[i] != y_true[i]) and str(y_pred[i]) != str(cl))
    accuracy = np.sum(TP) + np.sum(TN) / (np.sum(TP) + np.sum(TN) + np.sum(FP) + np.sum(FN))
    return accuracy


def accuracy_per_class(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    number_of_classes = len(set(y_pred))
    accuracy = [0]*number_of_classes
    assert(len(y_pred) == len(y_true))
    TP = [0] * number_of_classes
    FP = [0] * number_of_classes
    TN = [0] * number_of_classes
    FN = [0] * number_of_classes
    for cl in range(0, number_of_classes):
        for i in range(0, len(y_true)):
            TP[cl] += int(y_pred[i] == y_true[i] and str(y_pred[i]) == str(cl))
            FP[cl] += int(y_pred[i] != y_true[i] and str(y_pred[i]) == str(cl))
            TN[cl] += int(y_pred[i] == y_true[i] and str(y_pred[i]) != str(cl))
            FN[cl] += int(y_pred[i] != y_true[i] and str(y_pred[i]) != str(cl))
        accuracy[cl] = np.sum(TP[cl]) + np.sum(TN[cl]) / (np.sum(TP[cl]) + np.sum(TN[cl]) + np.sum(FP[cl]) + np.sum(FN[cl]))
    return accuracy


def r_squared1(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    sse = 0
    tse = 0
    y_true = list(y_true)
    assert (len(y_pred) == len(y_true))
    for r in range(0, len(y_true)):
        sse += (y_true[r] - y_pred[r]) ** 2
        tse += (y_true[r] - statistics.mean(y_true)) ** 2
    r2 = 1 - sse/tse
    return r2


def mse1(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    y_true = list(y_true)
    value = 0
    assert (len(y_pred) == len(y_true))
    for m in range(0, len(y_true)):
        value += (y_true[m] - y_pred[m]) ** 2
    mse_calc = value / len(y_true)
    return mse_calc


def mae1(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    y_true = list(y_true)
    value = 0
    assert (len(y_pred) == len(y_true))
    for m in range(0, len(y_true)):
        value += abs(y_true[m] - y_pred[m])
    mae_calc = value / len(y_true)
    return mae_calc
