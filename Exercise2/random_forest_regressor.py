import random
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

import regressor_tree


class RandomForestRegressor():

    def __init__(self, data, n_trees = 2, fitted_trees=[], max_features=2, min_samples_split=2, max_depth=2, height=0, 
                 left_child=None, right_child=None, flag="Internal", split_feature=None, 
                 split_value=None, prediction=None):
        self.data = data
        self.n_tress = n_trees
        self.fitted_trees = fitted_trees
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        self.height = height
        self.left_child  = left_child
        self.right_child = right_child
        self.flag = flag   # Internal or Leaf
        self.split_feature = split_feature
        self.split_value = split_value
        self.prediction = prediction


    def train(self):
        # create list of trees, store in attribute
        # think about bootstrapping to fit trees, as default use whole dataset to fit tree
        for i in range(self.n_tress):
            tree = regressor_tree.Node(data=self.data, max_features=self.max_features, min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.train()
            self.fitted_trees.append(tree)

        
    def predict(self, X_test):
        # iterate over list of trees, get prediction vector for test data, take average among trees
        predictions_each_tree = []
        for tree in self.fitted_trees:
            y_pred = tree.predict(X_test)
            predictions_each_tree.append(y_pred)

        # calculate predictions by taking mean of each tree
        mean_predictions = np.mean(predictions_each_tree, axis=0)
        return mean_predictions
        


if __name__ == "__main__":
    random.seed(6)

    # define column names for the dataset
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    # read data 
    data = pd.read_csv('auto-mpg.data', sep='\\s+', header=None, names=columns, quotechar='"')

    # TODO solve this for the following columns dropped for now
    data.drop(["horsepower", "car_name"], axis=1, inplace=True)

    X_train = data.drop('mpg', axis=1)
    y_train  = data["mpg"]

    rf = RandomForestRegressor(data=data, n_trees=2, max_features=1, min_samples_split=2, max_depth=2)
    rf.train()

    y_pred = rf.predict(X_test=X_train)

    print(rf.fitted_trees)