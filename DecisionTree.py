from collections import Counter
from toolset import partition, entropy, acc_gain
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None

    def build_tree(self, fvalue, label, depth=0):
        if len(set(label)) == 1 or depth == self.max_depth:
            return {"leaf": Counter(label).most_common(1)[0][0]}

        base_entropy = entropy(label)
        best_info_gain, best_feature = -1, None
        n_features = fvalue.shape[1]

        for f in range(n_features):
            gain = acc_gain(fvalue, label, f)
            if gain > best_info_gain:
                best_info_gain, best_feature = gain, f

        if best_feature is None:
            return {"leaf": Counter(label).most_common(1)[0][0]}

        (X_left, y_left), (X_right, y_right) = partition(fvalue, label, best_feature)

        return {
            "feature": best_feature,
            "entropy": base_entropy,
            "left": self.build_tree(np.array(X_left), np.array(y_left), depth + 1),
            "right": self.build_tree(np.array(X_right), np.array(y_right), depth + 1)
        }

    def fit(self, fvalue, label):
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
