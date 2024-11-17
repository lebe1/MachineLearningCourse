import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# Set random seed
RANDOM_SEED = 12

# Load dataset
df = pd.read_csv('../data/students_data.csv', delimiter=';')

# Extract the target variable 'Target' as y
y_student = df[['Target']]

# Extract all other columns as X (excluding 'Target')
X_student = df.drop('Target', axis=1)

X_student = X_student[['Application mode','Debtor','Tuition fees up to date','Gender','Scholarship holder','Age at enrollment','Curricular units 1st sem (approved)',
'Curricular units 1st sem (grade)',
'Curricular units 2nd sem (approved)',
'Curricular units 2nd sem (grade)',
'Curricular units 1st sem (evaluations)',
'Curricular units 2nd sem (evaluations)']]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_student, y_student, test_size=0.2, random_state=RANDOM_SEED)

# Set up the colnames
# colnames_numerical = df.loc[:, 'Curricular units 1st sem (credited)':'GDP'].columns.values

# colnames_numerical = np.append(colnames_numerical, ['Previous qualification (grade)','Admission grade', 'Age at enrollment'])

# colnames_categorical_1 = df.loc[:, 'Marital status':'Previous qualification'].columns.values

# colnames_categorical_2 = df.loc[:, 'Nacionality':"Father's occupation"].columns.values

# colnames_categorical = np.append(colnames_categorical_1, colnames_categorical_2)


colnames_numerical = [
    'Age at enrollment',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 2nd sem (evaluations)'
]

colnames_categorical = [
    'Application mode',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder'
]



# Create scaling pipelines for all continuous values
scale_pipe = make_pipeline(StandardScaler())
robust_scale_pipe = make_pipeline(RobustScaler())
min_max_scale_pipe = make_pipeline(MinMaxScaler())
log_pipe = make_pipeline(PowerTransformer())

# One-hot encode all categories represented by numbers (integer)
categorical_pipe = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown='ignore'))

transformer = ColumnTransformer(
    transformers=[
        #("scale", scale_pipe, colnames_numerical),
        ("robustscale", robust_scale_pipe, colnames_numerical),
        #("minmaxscaler", min_max_scale_pipe, colnames_numerical),
        #("log_transform", log_pipe, colnames_numerical),
        ("one_hot_encode", categorical_pipe, colnames_categorical),
    ]
)

# Explanation for GridSearchCV in Pipeline: https://stackoverflow.com/a/43366811/19932351
# We take 4 splits to divide the whole dataset into 5 splits with 0.2 percent for the test set
knn_pipe = Pipeline([("prep", transformer), 
                     ("knn", GridSearchCV(KNeighborsClassifier(),param_grid={'n_neighbors': [5, 10, 20, 30], 'weights': ['uniform', 'distance'], 'leaf_size': [2, 5, 10, 30, 50]}, cv=4, refit=True))])
random_forest_pipe = Pipeline([("prep", transformer), 
                                ("random_forest", GridSearchCV(RandomForestClassifier(random_state=RANDOM_SEED), param_grid={'n_estimators': [50, 100, 200], 'criterion': ['gini', 'entropy', 'log_loss'], 'max_features': ['sqrt', 'log2', None]}, cv=5, refit=True))])
mlp_pipe = Pipeline([("prep", transformer), ("mlp", GridSearchCV(MLPClassifier(random_state=RANDOM_SEED), param_grid={'hidden_layer_sizes': [50, 100, 200], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'learning_rate_init': [0.0001, 0.001, 0.01, 0.1], 'max_iter': [100, 200, 500] }, cv=5, refit=True))])

# Encode the target
le = LabelEncoder()
y_train = le.fit_transform(y_train.values.ravel())
y_test = le.fit_transform(y_test.values.ravel())


# Start measure time point
knn_start = time.time()

# Fit and predict knn
knn_pipe.fit(X_train, y_train)
knn_best_params = knn_pipe.named_steps['knn'].best_params_
predictions = knn_pipe.predict(X_test)
f1_knn = f1_score(y_test, predictions, average='macro')
acc_knn = accuracy_score(y_test, predictions)

knn_end = time.time()

# Fit and predict random forest
random_forest_pipe.fit(X_train, y_train)
random_forest_best_params = random_forest_pipe.named_steps['random_forest'].best_params_
predictions = random_forest_pipe.predict(X_test)
f1_rf = f1_score(y_test, predictions, average='macro')
acc_rf = accuracy_score(y_test, predictions)

random_forest_end = time.time()

# Fit and predict MLP
mlp_pipe.fit(X_train, y_train)
predictions = mlp_pipe.predict(X_test)
mlp_best_params = mlp_pipe.named_steps['mlp'].best_params_
f1_mlp = f1_score(y_test, predictions, average='macro')
acc_mlp = accuracy_score(y_test, predictions)

mlp_end = time.time()


# Open a file to write scores
with open("scores.txt", "w") as file:
    file.write(f"KNN\nF1_SCORE_MACRO: {round(f1_knn,2)}\nACCURACY: {round(acc_knn,2)}\n\n")
    file.write(f"KNN Execution time in s: {round(knn_end - knn_start,2)}\n\n")
    file.write(f"KNN Best params: {knn_best_params}\n\n")
    file.write(f"Random Forest\nF1_SCORE_MACRO: {round(f1_rf,2)}\nACCURACY: {round(acc_rf,2)}\n\n")
    file.write(f"Random Forest Execution time in s: {round(random_forest_end - knn_end,2)}\n\n")
    file.write(f"Random Forest Best params: {random_forest_best_params}\n\n")
    file.write(f"MLP\nF1_SCORE_MACRO: {round(f1_mlp,2)}\nACCURACY: {round(acc_mlp,2)}\n\n")
    file.write(f"MLP Execution time in s: {round(mlp_end - random_forest_end,2)}\n\n")
    file.write(f"MLP Best params: {mlp_best_params}\n\n")


