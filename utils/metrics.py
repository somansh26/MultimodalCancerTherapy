from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np


def compute_binary_classification_metrics(y_true, y_pred_logits):
    """
    Compute metrics for binary classification tasks.
    Args:
        y_true (array-like): True binary labels.
        y_pred_logits (array-like): Predicted logits (unnormalized probabilities).
    Returns:
        dict: Dictionary of metrics including ROC-AUC, F1-Score, Accuracy, Precision, and Recall.
    """
    y_pred = 1 / (1 + np.exp(-y_pred_logits))  # Apply sigmoid to logits
    y_pred_binary = (y_pred > 0.5).astype(int)

    metrics = {
        "ROC-AUC": roc_auc_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred_binary),
        "Accuracy": accuracy_score(y_true, y_pred_binary),
        "Precision": precision_score(y_true, y_pred_binary),
        "Recall": recall_score(y_true, y_pred_binary),
        "Confusion Matrix": confusion_matrix(y_true, y_pred_binary).tolist(),
    }
    return metrics


def compute_multiclass_classification_metrics(y_true, y_pred_logits):
    """
    Compute metrics for multi-class classification tasks.
    Args:
        y_true (array-like): True class labels.
        y_pred_logits (array-like): Predicted logits for each class.
    Returns:
        dict: Dictionary of metrics including Accuracy, Precision, Recall, and Confusion Matrix.
    """
    y_pred = np.argmax(y_pred_logits, axis=1)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro"),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro"),
        "F1-Score (Macro)": f1_score(y_true, y_pred, average="macro"),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def compute_c_index(y_true, y_pred):
    """
    Compute the concordance index (C-index) for survival prediction.
    Args:
        y_true (array-like): True survival times.
        y_pred (array-like): Predicted survival scores.
    Returns:
        float: C-index value.
    """
    n = 0
    h_sum = 0.0

    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                n += 1
                if y_pred[i] < y_pred[j] and y_true[i] < y_true[j]:
                    h_sum += 1
                elif y_pred[i] > y_pred[j] and y_true[i] > y_true[j]:
                    h_sum += 1
                elif y_pred[i] == y_pred[j]:
                    h_sum += 0.5

    return h_sum / n if n > 0 else 0.0



























