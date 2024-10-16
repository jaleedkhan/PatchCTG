import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score, accuracy_score

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
def classification_metrics(pred, true):

    # Calculate AUC
    auc = roc_auc_score(true, pred)
    
    # Compute ROC curve to find the optimal threshold using Youden's Index
    fpr, tpr, thresholds = roc_curve(true, pred)
    youdens_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youdens_index)]
    
    # Apply the optimal threshold to convert probabilities to binary labels
    pred_labels = (pred >= optimal_threshold).astype(int)
    true_labels = true.astype(int)
    
    # Calculate confusion matrix to extract TP, TN, FP, FN
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Sensitivity (Recall)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Specificity (FPR = 1 - Specificity)
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0          # Positive Predictive Value (Precision)
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0          # Negative Predictive Value
    f1 = f1_score(true_labels, pred_labels)                # F1 Score

    return accuracy, auc, sensitivity, specificity, ppv, npv, f1