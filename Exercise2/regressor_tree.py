import pandas as pd
import random
import numpy as np
from sklearn.metrics import mean_squared_error
import json

random.seed(6)

# Define column names for the dataset
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

data = pd.read_csv('Exercise2/auto-mpg.data', sep='\\s+', header=None, names=columns, quotechar='"')

X_train = data.drop('mpg', axis=1)


class RegressorTree():

    def __init__(self, max_features) -> None:
        self.max_features = max_features
    
    
    def select_random_feature(self, X_train):
        list_column_names = list(X_train.columns.values)
        feature_names = random.sample(list_column_names, self.max_features)
        
        dict_sorted_vectors = {}
        for name in feature_names:
            feature_vector = X_train[name].to_numpy().flatten()
            sorted_feature_vector = np.sort(feature_vector)
            dict_sorted_vectors[name] = list(sorted_feature_vector)

        return dict_sorted_vectors, feature_names

    def calculate_average_of_two_sample_pairs(self, series):
        average_list = []
        for count, row in enumerate(series):
            # Calculate average of each pair
            average = (row + series[count+1])/2

            average_list.append(average)
            
            # Condition to break for loop since count starts at 1 and stopping at second last element
            if count == (len(series) - 2):
                break

        return average_list


    def get_optimal_split_value(self, feature_name, value_list, df):
        sum_squared_residuals = {}
        for count, value in enumerate(value_list):
            # TODO fix condition especially for first and last value to be greater than or equals
            left_dataframe = df[df[feature_name] <= value]
            right_dataframe = df[df[feature_name] >= value]

            left_target_mean = left_dataframe['mpg'].mean()
            right_target_mean = right_dataframe['mpg'].mean()

            # Convert to numpy array
            left_mean_array = np.repeat(left_target_mean, len(left_dataframe))
            right_mean_array = np.repeat(right_target_mean, len(right_dataframe))

            # Calculate MSR
            right_rmse = mean_squared_error(right_dataframe['mpg'], right_mean_array)
            left_rmse = mean_squared_error(left_dataframe['mpg'], left_mean_array)

            # Store the sum squared residual with its value
            sum_squared_residuals[value] = left_rmse + right_rmse
            

        # Extract the minimum value of the sum_squared_residuals
        optimal_split_value = min(sum_squared_residuals, key=sum_squared_residuals.get)
        print(json.dumps(sum_squared_residuals,sort_keys=True, indent=4))
        print(optimal_split_value)


        return optimal_split_value
    
    ## Next steps
    # Marga: Implement predict function for MVP
    # Can: calculation of optimal split value for each feature
    # Leon: Construction of tree nodes



    

rtree = RegressorTree(max_features=2)
dict_features_sorted, feature_names = rtree.select_random_feature(X_train)
#average_list = rtree.calculate_average_of_two_sample_pairs(feature_series)
#rtree.split(feature_name, average_list, data)

print(json.dumps(dict_features_sorted,sort_keys=True, indent=4))

# TODO clean data input of X_train and _ytrain...