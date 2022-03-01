import numpy as np
import statistics


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

    tp, fp, fn, tn = 0, 0, 0, 0
    for yi_pred, yi_true in zip(y_pred, y_true):
        if yi_pred == yi_true == 1:
            tp += 1
        elif yi_pred == 1 and yi_true == 0:
            fp += 1
        elif yi_pred == 0 and yi_true == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * recall * precision / (recall + precision) if tp > 0 else 0
    accuracy = (y_pred == y_true).mean()
    return accuracy, precision, recall, f1

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    return (y_pred == y_true).mean()


def r_squared(y_pred, y_true):
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


def mse(y_pred, y_true):
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


def mae(y_pred, y_true):
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
    