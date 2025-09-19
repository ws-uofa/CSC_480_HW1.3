from collections import Counter
import numpy as np

def partition(fvalue, label, feature_index):
    false_idx = [i for i in range(len(fvalue)) if fvalue[i][feature_index] == 0]
    true_idx = [i for i in range(len(fvalue)) if fvalue[i][feature_index] == 1]
    return (fvalue[false_idx], label[false_idx]), (fvalue[true_idx], label[true_idx])

def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count/total) * np.log2(count/total) for count in counts.values())

def acc_gain(fvalue, label, feature_index):
        base_entropy = entropy(label)
        (X_false, y_false), (X_true, y_true) = partition(fvalue, label, feature_index)
        if len(y_false) == 0 or len(y_true) == 0:
            return -1
        new_entropy = (len(y_false) / len(label)) * entropy(y_false) + \
                      (len(y_true) / len(label)) * entropy(y_true)
        return base_entropy - new_entropy
