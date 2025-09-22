from collections import Counter
from toolset import partition, entropy, info_gain
import numpy as np


class DecisionTree:
    """
    A simple Decision Tree classifier implementation.

    Attributes:
    max_depth (int): The maximum depth the tree can grow.
    tree (dict): A nested dictionary structure representing the decision tree.
    """
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None

    def build_tree(self, fvalue, label, depth=0):
        """
        Build a decision tree recursively using entropy based information gain.

        Args:
        fvalue (ndarray): Feature values (2D array, shape: n_samples x n_features).
        label (list): Corresponding labels for the samples.
        depth (int): Current depth of the tree.

        Returns:
        dict: A nested dictionary representing the decision tree.
        """
        # If all labels are the same or max depth reached, return a leaf node
        if len(set(label)) == 1 or depth == self.max_depth:
            return {"leaf": Counter(label).most_common(1)[0][0]}

        base_entropy = entropy(label)
        best_info_gain, best_feature = -1, None
        n_features = fvalue.shape[1]

        # Find the best feature to split on based on information gain
        for f in range(n_features):
            gain = info_gain(fvalue, label, f)
            if gain > best_info_gain:
                best_info_gain, best_feature = gain, f
        # If no feature improves the split, return a leaf node
        if best_feature is None:
            return {"leaf": Counter(label).most_common(1)[0][0]}
        # Partition the data based on the best feature
        (X_false, y_false), (X_true, y_true) = partition(fvalue, label, best_feature)

        return {
            "feature": best_feature,
            "entropy": base_entropy,
            "left": self.build_tree(np.array(X_false), np.array(y_false), depth + 1),
            "right": self.build_tree(np.array(X_true), np.array(y_true), depth + 1)
        }

    def fit(self, fvalue, label):
        """
        Train the decision tree classifier on the given dataset.

        Args:
        fvalue (ndarray): Feature values.
        label (list): Labels.
        """
        self.tree = self.build_tree(np.array(fvalue), np.array(label))

    def print_tree(self, node=None, feature_names=None, depth=0):
        if node is None:
            node = self.tree
        prefix = "  " * depth
        if "leaf" in node:
            print(f"{prefix}Leaf → Predict {node['leaf']}")
        else:
            fname = feature_names[node["feature"]] if feature_names else f"f{node['feature']}"
            print(f"{prefix}Node: split on '{fname}', uncertainty score={node['entropy']:.3f}")
            print(f"{prefix} False:")
            self.print_tree(node["left"], feature_names, depth + 1)
            print(f"{prefix} True:")
            self.print_tree(node["right"], feature_names, depth + 1)
