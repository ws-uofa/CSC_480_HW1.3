# CSC_480_HW1.3

Decision Tree Practice for CSC 480 HW 1.3

## Overview

This repository contains a simple implementation of a decision tree classifier. The code demonstrates how to build, train, and visualize a decision tree using Python and NumPy.

## Files

- `DecisionTree.py`: Main decision tree class, including tree-building and printing logic.
- `toolset.py`: Utility functions for data partitioning, entropy, and information gain calculations.
- `main.py`: Script to load data, train the decision tree, and print its structure.
- `data.csv`: Data file (expects a CSV with features and a 'label' column).

## Usage

1. Make sure you have Python 3 and required packages (`pandas`, `numpy`) installed.
2. Prepare your dataset as `data.csv` in the same directory.
3. Run the main script:
   ```bash
   python main.py
   ```

## Example Output

Node: split on 'sys', uncertainty score=0.971
 False:
  Leaf → Predict 1
 True:
  Node: split on 'ai', uncertainty score=0.722
   False:
    Leaf → Predict 0
   True:
    Leaf → Predict 1