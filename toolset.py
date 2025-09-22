from collections import Counter
import numpy as np

def partition(fvalue, label, feature_index):
    """
    Split the dataset based on a binary feature.

    Args:
    fvalue (ndarray): Feature matrix (n_samples x n_features).
    label (ndarray): Labels corresponding to each sample.
    feature_index (int): Index of the feature used for partitioning.

    Returns:
    tuple: Two subsets of (features, labels) where feature == 0 and feature == 1.
    """
    false_idx = [i for i in range(len(fvalue)) if fvalue[i][feature_index] == 0]
    true_idx = [i for i in range(len(fvalue)) if fvalue[i][feature_index] == 1]
    return (fvalue[false_idx], label[false_idx]), (fvalue[true_idx], label[true_idx])

def entropy(y):
    """
    Compute entropy for a given label distribution.

    Args:
    y (list): Labels of the dataset.

    Returns:
    float: Entropy value.
    """
    counts = Counter(y)
    total = len(y)
    return -sum((count/total) * np.log2(count/total) for count in counts.values())

def info_gain(fvalue, label, feature_index):
    """
    Calculate the entropy-based information gain achieved by splitting on a feature.

    Args:
    fvalue (ndarray): Feature matrix.
    label (list): Labels.
    feature_index (int): Feature index used for splitting.

    Returns:
    float: Information gain value. Returns -1 if split is invalid.
    """
    base_entropy = entropy(label)
    (X_false, y_false), (X_true, y_true) = partition(fvalue, label, feature_index)

    # If one partition is empty, the split is invalid
    if len(y_false) == 0 or len(y_true) == 0:
        return -1

    # Weighted average of entropies after the split
    new_entropy = (len(y_false) / len(label)) * entropy(y_false) + (len(y_true) / len(label)) * entropy(y_true)
    return base_entropy - new_entropy
