import random
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split


class Node():

    def __init__(self, data, max_features=2, min_samples_split=2, max_depth=2, height=0, 
                 left_child=None, right_child=None, flag="Internal", split_feature=None, 
                 split_value=None, prediction=None) -> None:
        self.data = data
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


    def calculate_average_of_two_sample_pairs(self, dict_features_sorted):
        dict_averages_per_feature = {}

        # iterate over key-value-pairs of dict
        for feature_name, sorted_list in dict_features_sorted.items():
            
            dict_averages_per_feature[feature_name] = []
            for idx, curr_value in enumerate(sorted_list):
                # Calculate average of each pair of observations
                average_two_obs = (curr_value + sorted_list[idx + 1]) / 2

                dict_averages_per_feature[feature_name].append(average_two_obs)
                
                # Condition to break for loop since idx starts at and stopping at second last element
                if (idx == (len(sorted_list) - 2)):
                    break

        return dict_averages_per_feature

    def train(self):
        if ((len(self.data) >= self.min_samples_split) & (self.height < self.max_depth)):
            # split procedure
            # store results in new leftnode and rightnode

            data_left, data_right = train_test_split(self.data, test_size=0.5)

            self.left_child = Node(data=data_left, height=self.height + 1, max_features=self.max_features, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            self.right_child = Node(data=data_right, height=self.height + 1, max_features=self.max_features, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            self.left_child.train()
            self.right_child.train()
        else:
            self.flag = "Leaf"
            


    def predict(self):
        while (self.flag != "Leaf"):
            if (self.data[self.split_feature] < self.split_value):
                self = self.left_child
            else:
                self = self.right_child

        return self.prediction

    


class RegressorTree():

    def select_random_feature(self, X_train):
        list_column_names = list(X_train.columns.values)
        feature_names = random.sample(list_column_names, self.max_features)
        
        dict_sorted_vectors = {}
        for name in feature_names:
            feature_vector = X_train[name].to_numpy()
            sorted_feature_vector = np.sort(feature_vector)
            dict_sorted_vectors[name] = list(sorted_feature_vector)

        return dict_sorted_vectors, feature_names


    def calculate_average_of_two_sample_pairs(self, dict_features_sorted):
        dict_averages_per_feature = {}

        # iterate over key-value-pairs of dict
        for feature_name, sorted_list in dict_features_sorted.items():
            
            dict_averages_per_feature[feature_name] = []
            for idx, curr_value in enumerate(sorted_list):
                # Calculate average of each pair of observations
                average_two_obs = (curr_value + sorted_list[idx + 1]) / 2

                dict_averages_per_feature[feature_name].append(average_two_obs)
                
                # Condition to break for loop since idx starts at and stopping at second last element
                if (idx == (len(sorted_list) - 2)):
                    break

        return dict_averages_per_feature


    def get_optimal_split_value(self, dict_averages_per_feature, data):
        # dict structure: feature_name: [(split_val_1, ssr_1), (split_val_2, ssr_2), ...]
        dict_ssrs_per_feature = {}

        # iterate over key-value-pairs of dict
        for feature_name, avg_list in dict_averages_per_feature.items():

            dict_ssrs_per_feature[feature_name] = []
            for split_value in avg_list:
                # TODO fix condition especially for first and last value to be greater than or equals
                left_df  = data[data[feature_name] < split_value]
                right_df = data[data[feature_name] >= split_value]

                left_target_mean  = left_df["mpg"].mean()
                right_target_mean = right_df["mpg"].mean()

                # create numpy array with target mean for error metric
                left_mean_array  = np.repeat(left_target_mean, len(left_df))
                right_mean_array = np.repeat(right_target_mean, len(right_df))

                # calculate individual SSR (Sum Squared Residuals)
                left_ssr  = np.sum(np.square(left_df["mpg"] - left_mean_array))
                right_ssr = np.sum(np.square(right_df["mpg"] - right_mean_array))

                total_ssr = left_ssr + right_ssr
                tuple_split_val_ssr = (split_value, total_ssr, left_target_mean, right_target_mean)
                dict_ssrs_per_feature[feature_name].append(tuple_split_val_ssr)

        # find min SSR locally for each feature, then globally
        # list structure: [(feature_name_1, min_split_val_feature_1, corresponding_ssr), (feature_name_2, min_split_val_feature_2, corresponding_ssr), ...]
        list_min_ssr_per_feature = []
        for feature_name, tuple_list in dict_ssrs_per_feature.items():
            min_tuple  = min(tuple_list, key=lambda x: x[1])
            full_tuple = (feature_name, ) + min_tuple
            list_min_ssr_per_feature.append(full_tuple)

        # find min SSR globally
        tuple_optimal_split = min(list_min_ssr_per_feature, key=lambda x: x[2])

        return tuple_optimal_split
    

    def predict(self, optimal_split_value, X_test):
        #TO DO: Adapt to multiple features
        feature_name=optimal_split_value[0]
        split_value= optimal_split_value[1]
        left_prediction=optimal_split_value[3]
        right_prediction=optimal_split_value[4]

        y_pred_list=[]

        for index, row in X_test.iterrows():
            if row[feature_name]< split_value:
                y_pred=left_prediction
            else: 
                y_pred=right_prediction
            y_pred_list.append(y_pred)

        return y_pred_list
    



# rtree = RegressorTree(max_features=2)
# dict_features_sorted, feature_names = rtree.select_random_feature(X_train)
# dict_average_list = rtree.calculate_average_of_two_sample_pairs(dict_features_sorted)
# optimal_split_value = rtree.get_optimal_split_value(dict_average_list, data)
# y_pred=rtree.predict(optimal_split_value,X_train)


# print(optimal_split_value)
# print(set(y_pred))
# # print(json.dumps(dict_average_list,sort_keys=True, indent=4))

if __name__ == "__main__":
    random.seed(6)

    # define column names for the dataset
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    # read data 
    data = pd.read_csv('Exercise2/auto-mpg.data', sep='\\s+', header=None, names=columns, quotechar='"')

    X_train = data.drop('mpg', axis=1)
    y_train  = data["mpg"]

    root = Node(data=data, max_features=2, min_samples_split=2, max_depth=2, height=0)
    root.train()
    print(root.right_child.flag)


# TODO clean data input of X_train and _ytrain... --> "?" for some missing values

## Next steps
# Marga: Implement predict function for MVP
# Can: calculation of optimal split value for each feature
# Leon: Construction of tree nodes