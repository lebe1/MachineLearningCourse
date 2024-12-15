import numpy as np

# class representing single regression trees
class DecisionTreeRegressor:
    # attributes: max_depth to prevent overfitting; min_samples_split required to split a node
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    # constructs tree by calling _grow_tree method & stores it as dictionary in attribute tree
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    # make predictions for predictors by traversing the tree for each input sample using _traverse_tree
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    # recursive method to create tree
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # stops when maximum depth is reached or number of samples at node less than min_samples_split or all target values are identical
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.all(y == y[0]):
            # condition not met anymore, meaning its a leaf note, therefore returning target mean as prediction
            return np.mean(y)

        # when allowed to split, splits data into left & right subset based on best feature and theshold (found using _best_split)
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.mean(y)

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        # returns dictionary to store in attribute self.tree for all internal nodes
        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    # iterates over all features and thresholds to find split that minimizes the MSE
    # returns best features and corresponding best threshold for splitting
    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_mse = float("inf")

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                
                # calculates the variance-weighted MSE for each potential split using _calculate_mse method
                mse = self._calculate_mse(y[left_indices], y[right_indices])

                # update best_mse, feature, threshold if better solution found
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # calculates the variance-weighted MSE for each potential split
    def _calculate_mse(self, left_y, right_y):
        total = len(left_y) + len(right_y)
        left_mse = np.var(left_y) * len(left_y) if len(left_y) > 0 else 0
        right_mse = np.var(right_y) * len(right_y) if len(right_y) > 0 else 0
        return (left_mse + right_mse) / total

    # for each input sample, tree is traversed recursively based on feature values until a leaf node reached
    def _traverse_tree(self, x, node):
        # if node not dict, therefore (leaf), return current node
        if not isinstance(node, dict):
            return node

        # if node is dict, check if value of new observation is smaller/bigger than threshold to know if to traverse down left or right
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])


# class builds ensemble of DecisionTreeRegressor models & aggregates results
class RandomForestRegressor:
    # initialization with number of trees, max_depth, min_samples_split, max_features to consider for each split (currently unused)
    # stores all tree models in list
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    # creates n_estimators trees & appends them to list of trees
    # each tree trained on bootstrap sample (random sampling with replacement) using _bootstrap_sample method
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    # collect predictions vector for each tree --> results in matrix --> compute average to get final prediction vector
    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)


    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]



##########
# LLM: ChatGPT 4o
# Prompt: Your task is to implement a random forest regressor for predicting numeric values from scratch in Python. This means, you should not use the sklearn implementation. 

