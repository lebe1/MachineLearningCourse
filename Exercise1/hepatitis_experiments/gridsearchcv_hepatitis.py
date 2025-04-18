import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV



# Set random seed
RANDOM_SEED = 42

# Define column names for the dataset from the hepatitis.names file
columns = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']

df = pd.read_csv('../data/hepatitis/hepatitis.data', sep=',', header=None, names=columns)


# Extract the target variable 'Target' as y
y_hepatitis = df[['Class']]

# Extract all other columns as X (excluding 'Target')
X_hepatitis = df.drop('Class', axis=1)

#Preprocess whole train dataset by replacing ? for None value for imputation reasons
X_hepatitis = X_hepatitis.replace('?', np.nan)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_hepatitis, y_hepatitis, test_size=0.2, shuffle=True, random_state=RANDOM_SEED)


print("After split",X_train.isna().sum().sum())
print("After split",X_test.isna().sum().sum())

# Split up the column names into numerical and categorical attributes 
colnames_numerical = ["AGE", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME"]

colnames_categorical = ['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 
            'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']

# Impute missing values
cat_impute_pipe = make_pipeline(SimpleImputer(strategy='most_frequent', missing_values=np.nan))
num_impute_pipe = make_pipeline(SimpleImputer(strategy='median', missing_values=np.nan))


# Scale and log transform all continuous values
scale_pipe = make_pipeline(StandardScaler())
log_pipe = make_pipeline(PowerTransformer())


impute = ColumnTransformer(
    transformers=[
        ("impute_categories", cat_impute_pipe, colnames_categorical),
        ("impute_numerics", num_impute_pipe, colnames_numerical),
    ]
)

# Impute both train sets before inserting to model to prevent error for models not able to handle NaNs
imputed_X_train = pd.DataFrame(impute.fit_transform(X_train))
imputed_X_test = pd.DataFrame(impute.fit_transform(X_test))

# sk = StandardScaler()
# scaled_X_train = pd.DataFrame(sk.fit_transform(imputed_X_train))
# scaled_X_test = pd.DataFrame(sk.fit_transform(imputed_X_test))

# print("After scale",scaled_X_train.isna().sum().sum())

# Convert target value to numpy array
y_test = y_test.values.flatten()
y_train = y_train.values.flatten()

param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["minkowski", "euclidean", "manhattan"]
}

param_grid_rf = {
    "n_estimators": [100, 200, 500],
    "max_features": ["sqrt", "log2", 0.2, 0.5],
    "max_depth": [None, 10, 25, 40],
    "min_samples_leaf": [1, 5]
}

param_grid_mlp = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
    "activation": ["relu", "tanh", "logistic"],
    "solver": ["adam", "sgd"],
    "learning_rate_init": [0.01, 0.001, 0.0001]
}


# Set pipelines
knn_model = GridSearchCV(KNeighborsClassifier(),param_grid=param_grid_knn, cv=5, scoring='f1')
random_forest_model = GridSearchCV(RandomForestClassifier(random_state=RANDOM_SEED), param_grid=param_grid_rf, cv=5, scoring='f1')
mlp_model = GridSearchCV(MLPClassifier(random_state=RANDOM_SEED), param_grid=param_grid_mlp, cv=5, scoring='f1')


# Start measure time point
knn_start = time.time()

# Start measure time point
knn_start = time.time()

# Fit and predict knn
knn_model.fit(imputed_X_train, y_train)
knn_best_params = knn_model.best_params_
predictions = knn_model.predict(imputed_X_test)
f1_knn = f1_score(y_test, predictions, average='binary')
acc_knn = accuracy_score(y_test, predictions)

knn_end = time.time()

# Fit and predict random forest
random_forest_model.fit(imputed_X_train, y_train)
random_forest_best_params = random_forest_model.best_params_
predictions = random_forest_model.predict(imputed_X_test)
f1_rf = f1_score(y_test, predictions, average='binary')
acc_rf = accuracy_score(y_test, predictions)

random_forest_end = time.time()

# Fit and predict MLP
mlp_model.fit(imputed_X_train, y_train)
predictions = mlp_model.predict(imputed_X_test)
mlp_best_params = mlp_model.best_params_
f1_mlp = f1_score(y_test, predictions, average='binary')
acc_mlp = accuracy_score(y_test, predictions)

mlp_end = time.time()


# Open a file to write scores
with open("scores.txt", "w") as file:
    file.write(f"KNN\nF1_SCORE_BINARY: {round(f1_knn,2)}\nACCURACY: {round(acc_knn,2)}\n\n")
    file.write(f"KNN Execution time in s: {round(knn_end - knn_start,2)}\n\n")
    file.write(f"KNN Best params: {knn_best_params}\n\n")
    file.write(f"Random Forest\nF1_SCORE_BINARY: {round(f1_rf,2)}\nACCURACY: {round(acc_rf,2)}\n\n")
    file.write(f"Random Forest Execution time in s: {round(random_forest_end - knn_end,2)}\n\n")
    file.write(f"Random Forest Best params: {random_forest_best_params}\n\n")
    file.write(f"MLP\nF1_SCORE_BINARY: {round(f1_mlp,2)}\nACCURACY: {round(acc_mlp,2)}\n\n")
    file.write(f"MLP Execution time in s: {round(mlp_end - random_forest_end,2)}\n\n")
    file.write(f"MLP Best params: {mlp_best_params}\n\n")

