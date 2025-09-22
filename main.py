import pandas as pd
from DecisionTree import DecisionTree

# Load dataset from CSV file
data = pd.read_csv('data.csv')

# Separate features (X) and labels (y)
fvalue = data.drop(columns=["label"]).values
labels = data["label"].values
features = data.drop(columns=["label"]).columns.tolist()


# Initialize the Decision Tree with a maximum depth of 2
tree = DecisionTree(max_depth=2)

# Train the model
tree.fit(fvalue, labels)


# Print the structure of the trained tree
tree.print_tree(feature_names=features)
