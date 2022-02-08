import numpy as np


def binary_classification_metrics(y_pred, y_true):
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
    metrics['accuracy'] = multiclass_accuracy(y_pred, y_true)
    return metrics

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    assert(len(y_pred) == len(y_true))
    TP = [0] * 10
    FP = [0] * 10
    TN = [0] * 10
    FN = [0] * 10
    for cl in range(0, 10):
        for i in range(0, len(y_true)):
            TP[cl] += int((y_pred[i] == y_true[i]) & (y_pred[i] == str(cl)))
            FP[cl] += int((y_pred[i] != y_true[i]) & (y_pred[i] == str(cl)))
            TN[cl] += int((y_pred[i] == y_true[i]) & (y_pred[i] != str(cl)))
            FN[cl] += int((y_pred[i] != y_true[i]) & (y_pred[i] != str(cl)))
    accuracy = np.sum(TP) + np.sum(TN) / (np.sum(TP) + np.sum(TN) + np.sum(FP) + np.sum(FN))
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """
    pass
    