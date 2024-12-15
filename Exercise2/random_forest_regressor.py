import random
import pandas as pd
import numpy as np


import regressor_tree


class RandomForestRegressor():

    def __init__(self, n_trees = 5, fitted_trees=[], max_features=1, min_samples_split=2, max_depth=5, height=0, 
                 left_child=None, right_child=None, flag="Internal", split_feature=None, 
                 split_value=None, prediction=None, random_state=None):
      
        self.n_trees = n_trees
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
        self.random_state = random_state




    def train(self, X_data, y_data):
        # set seed for reproducibility and reliable randomness
        if self.random_state is not None:
            random.seed(self.random_state)
            random_list = [random.randint(0, 2**32 - 1) for _ in range(self.n_trees)]
        else:
            random_list = [None] * self.n_trees


        # Create list of trees and store in the fitted_trees attribute
        for i in range(self.n_trees):
            tree = regressor_tree.Node(
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                random_state=random_list[i]
            )
            tree.train(X_data, y_data)
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
    # define column names for the dataset
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    # read data 
    data = pd.read_csv('auto-mpg.data', sep='\\s+', header=None, names=columns, quotechar='"')

    data.drop(["horsepower", "car_name"], axis=1, inplace=True)

    X_train = data.drop('mpg', axis=1)
    y_train  = data["mpg"]

    rf = RandomForestRegressor(n_trees=2, max_features=1, min_samples_split=2, max_depth=2, random_state=42)
    rf.train(X_train, y_train)

    y_pred = rf.predict(X_test=X_train)

    print(rf.fitted_trees)