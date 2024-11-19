import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

# Set random seed
RANDOM_SEED = 42

# Load dataset
df = pd.read_csv('../data/students_data.csv', delimiter=';')

# Extract the target variable 'Target' as y
y_student = df[['Target']]

# Extract all other columns as X (excluding 'Target')
X_student = df.drop('Target', axis=1)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_student, y_student, test_size=0.2, random_state=RANDOM_SEED)

# Set up the colnames
colnames_numerical = df.loc[:, 'Curricular units 1st sem (credited)':'GDP'].columns.values

colnames_numerical = np.append(colnames_numerical, ['Previous qualification (grade)','Admission grade', 'Age at enrollment'])

colnames_categorical_1 = df.loc[:, 'Marital status':'Previous qualification'].columns.values

colnames_categorical_2 = df.loc[:, 'Nacionality':"Father's occupation"].columns.values

colnames_categorical = np.append(colnames_categorical_1, colnames_categorical_2)


# Create scaling pipelines for all continuous values
scale_pipe = make_pipeline(StandardScaler())
log_pipe = make_pipeline(PowerTransformer())

# One-hot encode all categories represented by numbers (integer)
categorical_pipe = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown='ignore'))

transformer = ColumnTransformer(
    transformers=[
        #("scale", scale_pipe, colnames_numerical),
        #("log_transform", log_pipe, colnames_numerical[13]),
        ("one_hot_encode", categorical_pipe, colnames_categorical),
    ]
)

knn_pipe = Pipeline([("prep", transformer), ("knn", KNeighborsClassifier())])
random_forest_pipe = Pipeline([("prep", transformer), ("random_forest", RandomForestClassifier(random_state=RANDOM_SEED))])
mlp_pipe = Pipeline([("prep", transformer), ("mlp", MLPClassifier(random_state=RANDOM_SEED))])

# Encode the target
le = LabelEncoder()
y_train = le.fit_transform(y_train.values.ravel())
y_test = le.fit_transform(y_test.values.ravel())

# Define pipelines
knn_pipe = Pipeline([("prep", transformer), ("knn", KNeighborsClassifier())])
random_forest_pipe = Pipeline([("prep", transformer), ("random_forest", RandomForestClassifier(random_state=RANDOM_SEED))])
mlp_pipe = Pipeline([("prep", transformer), ("mlp", MLPClassifier(random_state=RANDOM_SEED))])

# KNN configurations
knn_configs = [
    ("Default (n=5)", {"knn__n_neighbors": 5}),
    ("Small n (n=1)", {"knn__n_neighbors": 1}),
    ("Big n (n=10)", {"knn__n_neighbors": 10}),
    ("Biggest n (n=50)", {"knn__n_neighbors": 50})
]

# Random Forest configurations
rf_configs = [
    ("Default (n_estimators=100)", {"random_forest__n_estimators": 100}),
    ("n_estimators=10", {"random_forest__n_estimators": 10}),
    ("n_estimators=50", {"random_forest__n_estimators": 50}),
    ("n_estimators=1000", {"random_forest__n_estimators": 1000})
]

# MLP configurations
mlp_configs = [
    ("Default (hidden_layer_sizes=(100,))", {"mlp__hidden_layer_sizes": (100,)}),
    ("hidden_layer_sizes=(10,)", {"mlp__hidden_layer_sizes": (10,)}),
    ("hidden_layer_sizes=(50, 50)", {"mlp__hidden_layer_sizes": (50, 50)}),
    ("hidden_layer_sizes=(100, 100)", {"mlp__hidden_layer_sizes": (100, 100)})
]

# Helper function to fit, predict, and evaluate
def evaluate_pipeline(pipe, configs, X_train, X_test, y_train, y_test, start_time):
    results = []
    for name, params in configs:
        pipe.set_params(**params)  
        pipe.fit(X_train, y_train)  
        y_pred = pipe.predict(X_test)  
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        results.append((name, acc, f1, time.time() - start_time)) 
    return results

# Evaluate KNN
knn_start = time.time()
knn_results = evaluate_pipeline(knn_pipe, knn_configs, X_train, X_test, y_train, y_test, knn_start)
knn_end = time.time()

# Evaluate Random Forest
rf_start = knn_end
rf_results = evaluate_pipeline(random_forest_pipe, rf_configs, X_train, X_test, y_train, y_test, rf_start)
rf_end = time.time()

# Evaluate MLP
mlp_start = rf_end
mlp_results = evaluate_pipeline(mlp_pipe, mlp_configs, X_train, X_test, y_train, y_test, mlp_start)
mlp_end = time.time()

# Write results to file
with open("scores.txt", "w") as file:
    # Write KNN results
    file.write("KNN Results:\n")
    for name, acc, f1, exec_time in knn_results:
        file.write(f"  {name} - Accuracy: {round(acc, 2)}, F1 Score: {round(f1, 2)}, Execution Time: {round(exec_time, 2)} seconds\n")
    file.write("\n")

    # Write Random Forest results
    file.write("Random Forest Results:\n")
    for name, acc, f1, exec_time in rf_results:
        file.write(f"  {name} - Accuracy: {round(acc, 2)}, F1 Score: {round(f1, 2)}, Execution Time: {round(exec_time, 2)} seconds\n")
    file.write("\n")

    # Write MLP results
    file.write("MLP Results:\n")
    for name, acc, f1, exec_time in mlp_results:
        file.write(f"  {name} - Accuracy: {round(acc, 2)}, F1 Score: {round(f1, 2)}, Execution Time: {round(exec_time, 2)} seconds\n")
    file.write("\n")




