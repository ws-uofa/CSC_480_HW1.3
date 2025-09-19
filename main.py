import pandas as pd
from DecisionTree import DecisionTree

data = pd.read_csv('data.csv')
fvalue = data.drop(columns=["label"]).values
labels = data["label"].values
features = data.drop(columns=["label"]).columns.tolist()

tree = DecisionTree(max_depth=2)
tree.fit(fvalue, labels)
tree.print_tree(feature_names=features)
