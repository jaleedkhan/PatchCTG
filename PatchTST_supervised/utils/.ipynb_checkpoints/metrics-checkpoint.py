import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    return mae, mse, rmse, mape, mspe, rse, corr

# Binary classification metrics
def accuracy(pred, true):
    pred_labels = (pred > 0.5).astype(int)
    true_labels = true.astype(int)
    return accuracy_score(true_labels, pred_labels)

def precision(pred, true):
    pred_labels = (pred > 0.5).astype(int)
    true_labels = true.astype(int)
    return precision_score(true_labels, pred_labels)

def recall(pred, true):
    pred_labels = (pred > 0.5).astype(int)
    true_labels = true.astype(int)
    return recall_score(true_labels, pred_labels)

def f1(pred, true):
    pred_labels = (pred > 0.5).astype(int)
    true_labels = true.astype(int)
    return f1_score(true_labels, pred_labels)

def auc(pred, true):
    true_labels = true.astype(int)
    return roc_auc_score(true_labels, pred)

def classification_metrics(pred, true):
    acc = accuracy(pred, true)
    prec = precision(pred, true)
    rec = recall(pred, true)
    f1_sc = f1(pred, true)
    auc_sc = auc(pred, true)
    return acc, prec, rec, f1_sc, auc_sc

