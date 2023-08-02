from __future__ import annotations

import numpy as np


def _check_nparray(array):
    return array if isinstance(array, np.ndarray) else np.array(array)


def _sum_positive(y):
    return np.sum(y)


def _sum_negative(y):
    return np.sum(y == 0)


def _true_positive(y_true, y_pred):
    return np.sum(((y_pred == 1) & (y_true == 1)).astype(int))


def _false_positive(y_true, y_pred):
    return np.sum(((y_pred == 1) & (y_true == 0)).astype(int))


def _true_negative(y_true, y_pred):
    return np.sum(((y_pred == 0) & (y_true == 0)).astype(int))


def _false_negative(y_true, y_pred):
    return np.sum(((y_pred == 0) & (y_true == 1)).astype(int))


def confusion_matrix(y_true, y_pred):
    uncertainty = 1.0 / np.sqrt(y_true.size)

    return np.array(
        [
            [[_true_positive(y_true, y_pred), uncertainty], [_false_negative(y_true, y_pred), uncertainty]],
            [[_false_positive(y_true, y_pred), uncertainty], [_true_negative(y_true, y_pred), uncertainty]],
        ]
    )


def true_positive_rate(y_true, y_pred):
    y_true = _check_nparray(y_true)
    y_pred = _check_nparray(y_pred)

    tpr = _true_positive(y_true, y_pred) / _sum_positive(y_true)
    tpr_uncertainty = 1.0 / np.sqrt(y_true.size)

    return tpr, tpr_uncertainty


def true_negative_rate(y_true, y_pred):
    y_true = _check_nparray(y_true)
    y_pred = _check_nparray(y_pred)

    tnr = _true_negative(y_true, y_pred) / _sum_negative(y_true)
    tnr_uncertainty = 1.0 / np.sqrt(y_true.size)

    return tnr, tnr_uncertainty


def false_positive_rate(y_true, y_pred):
    y_true = _check_nparray(y_true)
    y_pred = _check_nparray(y_pred)

    fpr = _false_positive(y_true, y_pred) / _sum_positive(y_true)
    fpr_uncertainty = 1.0 / np.sqrt(y_true.size)

    return fpr, fpr_uncertainty


def false_negative_rate(y_true, y_pred):
    y_true = _check_nparray(y_true)
    y_pred = _check_nparray(y_pred)

    fnr = _false_negative(y_true, y_pred) / _sum_negative(y_true)
    fnr_uncertainty = 1.0 / np.sqrt(y_true.size)

    return fnr, fnr_uncertainty


def precision(y_true, y_pred):
    y_true = _check_nparray(y_true)
    y_pred = _check_nparray(y_pred)

    tp = _true_positive(y_true, y_pred)
    fp = _false_positive(y_true, y_pred)

    if tp == 0 and fp == 0:
        return 0, 0

    precision = tp / (tp + fp)
    precision_uncertainty = 1.0 / np.sqrt(y_true.size)

    return precision, precision_uncertainty


def negative_precision(y_true, y_pred):
    y_true = _check_nparray(y_true)
    y_pred = _check_nparray(y_pred)

    tn = _true_negative(y_true, y_pred)
    fn = _false_negative(y_true, y_pred)

    if tn == 0 and fn == 0:
        return 0, 0

    negative_precision = tn / (tn + fn)
    negative_precision_uncertainty = 1.0 / np.sqrt(y_true.size)

    return negative_precision, negative_precision_uncertainty


def recall(y_true, y_pred):
    return true_positive_rate(y_true, y_pred)


def fbeta_score(y_true, y_pred, beta):
    _precision, uncertainty = precision(y_true, y_pred)
    _recall, _ = recall(y_true, y_pred)

    return (1 + beta**2) / (beta**2 * _precision + _recall), uncertainty


def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)
