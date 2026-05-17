import numpy as np

def f1_score_from_confusion_matrix(cm):
    """
    cm: list of lists atau numpy array (n_classes x n_classes)
    return: list of f1-score per class (urutan sesuai baris/kolom)
    """
    cm = np.array(cm)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1.tolist()