import random
import pandas as pd
import numpy as np



class Node():
    """
    Represents a single node which is recursively extended with children nodes to form a decision tree for a Random Forest Regressor.
    """

    def __init__(self, max_features=1, min_samples_split=2, max_depth=5, height=0, 
                 left_child=None, right_child=None, flag="Internal", split_feature=None, 
                 split_value=None, prediction=None, random_state=None, node_index=1) -> None:

        """
        Initializes instance of class Node with the provided parameters.

        Args:
            max_features (int): The maximum number of features to consider when splitting.
            min_samples_split (int): The minimum number of samples required to split a node.
            max_depth (int): The maximum depth of the tree. None means no limit.
            height (int): The height of a node in the tree.
            left_child (Node): Reference to the left child node.
            right_child (Node): Reference to the right child node.
            flag (str): Indicates whether the node is "Internal" or a "Leaf".
            split_feature (str): The feature used for splitting at an internal node.
            split_value (float): The value used to split the data at an internal node.
            prediction (float): The prediction value if the node is a leaf.
            random_state (int or None): Random seed for reproducibility.
            node_index (int): The index of the node within the tree.
        """

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
        self.node_index = node_index

        self.X_data = None
        self.y_data = None
    

    def get_optimal_split_value(self, dict_averages_per_feature):
        """
        Identifies the optimal feature and value to split the data at an internal node.

        Args:
            dict_averages_per_feature (dict): A dictionary mapping feature names to a list of average split values for potential splits.

        Returns:
            optimal_split (tuple): A tuple containing the best feature to split on and the corresponding split value.
        """

        dict_ssrs_per_feature = {}

        # iterate over key-value-pairs of dict
        for feature_name, avg_list in dict_averages_per_feature.items():

            dict_ssrs_per_feature[feature_name] = []
            for split_value in avg_list:
                # Edge case proven in case one dataframe is empty
                left_df  = self.y_data[self.X_data[feature_name] < split_value]
                right_df = self.y_data[self.X_data[feature_name] >= split_value]

                left_target_mean  = left_df.mean()
                right_target_mean = right_df.mean()

                # create numpy array with target mean for error metric
                left_mean_array  = np.repeat(left_target_mean, len(left_df))
                right_mean_array = np.repeat(right_target_mean, len(right_df))

                # calculate individual SSR (Sum Squared Residuals)
                left_ssr  = np.sum(np.square(left_df - left_mean_array))
                right_ssr = np.sum(np.square(right_df - right_mean_array))

                total_ssr = left_ssr + right_ssr
                tuple_split_val_ssr = {"Split value": split_value, "Total SSR": total_ssr}
                dict_ssrs_per_feature[feature_name].append(tuple_split_val_ssr)

        # find min SSR locally for each feature, then globally
        # list structure: [(feature_name_1, min_split_val_feature_1, corresponding_ssr), (feature_name_2, min_split_val_feature_2, corresponding_ssr), ...]
        dict_best_values_per_feature = {}
        for feature_name, dictionary_list in dict_ssrs_per_feature.items():
            min_ssr = dictionary_list[0]["Total SSR"]
            best_split_value = dictionary_list[0]["Split value"]
            for dictionary in dictionary_list:
                if dictionary["Total SSR"] < min_ssr:
                    min_ssr = dictionary["Total SSR"]
                    best_split_value = dictionary["Split value"]

            # Add feature name and best split values
            dict_best_values_per_feature[feature_name] = {"Best split": best_split_value, "Min SSR": min_ssr}

        # find min SSR, best split value and best feature globally over every feature
        global_min_ssr = dict_best_values_per_feature[str(next(iter(dict_best_values_per_feature)))]["Min SSR"]
        global_best_split_value = dict_best_values_per_feature[next(iter(dict_best_values_per_feature))]["Best split"]

        # Get first key of dictionary as placeholder, which is the first feature name
        best_feature = list(dict_best_values_per_feature.keys())[0]

        for feature_name, dictionary in dict_best_values_per_feature.items():
                
            if dictionary["Min SSR"] < global_min_ssr:
                global_min_ssr = dictionary["Min SSR"]
                global_best_split_value = dictionary["Best split"]
                best_feature = feature_name

        optimal_split = (best_feature, global_best_split_value)

        return optimal_split


    def train(self, X_data, y_data):
        """
        Trains the Regressor Tree recursively by splitting the data and creating child nodes.

        Args:
            X_data: The predictors to be used for training.
            y_data: The target variable corresponding to X_data.
        """

        # Set data instances
        self.X_data = X_data
        self.y_data = y_data

        # This huge if-statement could not be improved due to the ability to set max_depth to None
        # First condition checks that data is greater or equals than the min samples split and the tree height is lower than the max depth
        # Second condition has the same first check on min samples split but checks for max_depth set to None so the tree grows until the min sample split is true
        if (
            (len(self.X_data) >= self.min_samples_split and (self.max_depth is None or self.height < self.max_depth))
            or
            (len(self.X_data) >= self.min_samples_split and self.max_depth is None)
        ):               
            

            # store results in new leftnode and rightnode
            dict_averages_per_feature = self.get_average_values_per_feature()
            # Insert split value for current node
            self.split_feature, self.split_value = self.get_optimal_split_value(dict_averages_per_feature)


            # Split data according to split value
            data_left_X  = self.X_data[self.X_data[self.split_feature] < self.split_value]
            data_left_y = self.y_data[self.X_data[self.split_feature] < self.split_value]
            
            data_right_X = self.X_data[self.X_data[self.split_feature] >= self.split_value]
            data_right_y = self.y_data[self.X_data[self.split_feature] >= self.split_value]


            self.left_child = Node(height=self.height + 1, max_features=self.max_features, max_depth=self.max_depth, min_samples_split=self.min_samples_split, random_state=self.random_state, node_index=(2 * self.node_index))
            self.right_child = Node(height=self.height + 1, max_features=self.max_features, max_depth=self.max_depth, min_samples_split=self.min_samples_split, random_state=self.random_state, node_index=(2 * self.node_index)+1)

            self.left_child.train(data_left_X, data_left_y)
            self.right_child.train(data_right_X, data_right_y)
        else:
            # Get the prediction value by the mean of the remaining target data (y_data)
            self.prediction = self.y_data.mean()
            self.flag = "Leaf"

            


    def predict(self, X_test):
        """
        Iterates over each test set observation and predicts target value by traversing the tree until leaf node.

        Args:
            X_test: The predictor data for which predictions should be made.

        Returns:
            y_pred (list): List of predicted target values corresponding to X_test.
        """

        y_pred = []
        for index, row in X_test.iterrows():
            while (self.flag != "Leaf"):
                if (row[self.split_feature] < self.split_value):
                    self = self.left_child
                else:
                    self = self.right_child

            y_pred.append(self.prediction)
        
        return y_pred

    
    def get_average_values_per_feature(self):
        """
        Randomly selects a subset of features based on the max_features parameter and calculates the mean of consecutive sorted values for potential splits for each feature

        Returns:
            dict_averages_per_feature: A dictionary where keys are feature names and values are lists of average split values.
        """

        # Set random list based on max_depth and random_state
        if self.random_state is not None:
            random.seed(self.random_state)
            if self.max_depth is not None:
                random_list = [random.randint(0, 2**32 - 1) for _ in range(2**(self.max_depth + 1) - 1)]
            else:
                # else statement to enable setting max_depth to None
                random_list = [random.randint(0, 2**32 - 1) for _ in range(2**(100 + 1) - 1)] 
            random.seed(random_list[self.node_index])
        
        list_column_names = list(self.X_data.columns.values)

        if (self.max_features > len(self.X_data.columns)):
            # set max_features to number of predictors if it is initialized too large
            self.max_features = len(self.X_data.columns)
        
        feature_names = random.sample(list_column_names, self.max_features)

        
        dict_sorted_vectors = {}
        for name in feature_names:
            feature_vector = self.X_data[name].to_numpy()
            sorted_feature_vector = np.sort(feature_vector)
            dict_sorted_vectors[name] = list(sorted_feature_vector)

 
        dict_averages_per_feature = {}

        # iterate over key-value-pairs of dict
        for feature_name, sorted_list in dict_sorted_vectors.items():
            
            dict_averages_per_feature[feature_name] = []
            for idx, curr_value in enumerate(sorted_list):
                # Calculate average of each pair of observations
                average_two_obs = (curr_value + sorted_list[idx + 1]) / 2

                dict_averages_per_feature[feature_name].append(average_two_obs)
                
                # Condition to break for loop since idx starts at and stopping at second last element
                if (idx == (len(sorted_list) - 2)):
                    break

        return dict_averages_per_feature    


if __name__ == "__main__":
    # define column names for the dataset
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    # read data 
    data = pd.read_csv('auto-mpg.data', sep='\\s+', header=None, names=columns, quotechar='"')

    data.drop(["horsepower", "car_name"], axis=1, inplace=True)

    X_train = data.drop('mpg', axis=1)
    y_train  = data["mpg"]

    root = Node(max_features=1, min_samples_split=2, max_depth=4, height=0, random_state=42)
    root.train(X_train, y_train)
    print("Root flag: ", root.flag)
    print("Left left split value: ", root.left_child.left_child.split_value)
    
    print("Left left prediciton value: ", root.left_child.left_child.prediction)
    print("Left left flag; ", root.left_child.left_child.flag)

