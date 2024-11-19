import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
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
from sklearn.impute import SimpleImputer


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

# Set pipelines
knn_default = KNeighborsClassifier(n_neighbors=5)       # 5 is the default value for n_neighbors
knn_small_n = KNeighborsClassifier(n_neighbors=1)
knn_big_n   = KNeighborsClassifier(n_neighbors=10)
knn_biggest_n   = KNeighborsClassifier(n_neighbors=50)
random_forest_model = RandomForestClassifier(random_state=RANDOM_SEED)
mlp_model = MLPClassifier(random_state=RANDOM_SEED)


# Start measure time point
knn_start = time.time()

knn_default.fit(imputed_X_train, y_train)
knn_small_n.fit(imputed_X_train, y_train)
knn_big_n.fit(imputed_X_train, y_train)
knn_biggest_n.fit(imputed_X_train, y_train)

y_pred_default = knn_default.predict(imputed_X_test)
y_pred_small_n = knn_small_n.predict(imputed_X_test)
y_pred_big_n   = knn_big_n.predict(imputed_X_test)
y_pred_biggest_n   = knn_biggest_n.predict(imputed_X_test)


acc_default = accuracy_score(y_test, y_pred_default, normalize=True)
f1_default  = f1_score(y_test, y_pred_default, average="binary")

acc_small_n = accuracy_score(y_test, y_pred_small_n, normalize=True)
f1_small_n  = f1_score(y_test, y_pred_small_n, average="binary")

acc_big_n = accuracy_score(y_test, y_pred_big_n, normalize=True)
f1_big_n  = f1_score(y_test, y_pred_big_n, average="binary")

acc_biggest_n = accuracy_score(y_test, y_pred_biggest_n, normalize=True)
f1_biggest_n  = f1_score(y_test, y_pred_biggest_n, average="binary")


knn_end = time.time()

rf_default = RandomForestClassifier(n_estimators=100, random_state=42)       # 100 is the default value for n_estimators
rf_10      = RandomForestClassifier(n_estimators=10, random_state=42)
rf_50      = RandomForestClassifier(n_estimators=50, random_state=42)
rf_1000    = RandomForestClassifier(n_estimators=1000, random_state=42)

rf_default.fit(imputed_X_train, y_train)
rf_10.fit(imputed_X_train, y_train)
rf_50.fit(imputed_X_train, y_train)
rf_1000.fit(imputed_X_train, y_train)

y_pred_default  = rf_default.predict(imputed_X_test)
y_pred_10       = rf_10.predict(imputed_X_test)
y_pred_50      = rf_50.predict(imputed_X_test)
y_pred_1000      = rf_1000.predict(imputed_X_test)

acc_default = accuracy_score(y_test, y_pred_default, normalize=True)
f1_default  = f1_score(y_test, y_pred_default, average="binary")

acc_10 = accuracy_score(y_test, y_pred_10, normalize=True)
f1_10  = f1_score(y_test, y_pred_10, average="binary")

acc_50 = accuracy_score(y_test, y_pred_50, normalize=True)
f1_50  = f1_score(y_test, y_pred_50, average="binary")

acc_1000 = accuracy_score(y_test, y_pred_1000, normalize=True)
f1_1000  = f1_score(y_test, y_pred_1000, average="binary")

# Fit and predict random forest
random_forest_model.fit(imputed_X_train, y_train)
random_forest_predictions = random_forest_model.predict(imputed_X_test)
f1_rf = f1_score(y_test, random_forest_predictions, average='binary')
acc_rf = accuracy_score(y_test, random_forest_predictions)

random_forest_end = time.time()

mlp_default = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)      # (100,) is the default value for hidden_layer_sizes
mlp_10      = MLPClassifier(hidden_layer_sizes=(10,), random_state=42) 
mlp_50_50     = MLPClassifier(hidden_layer_sizes=(50, 50), random_state=42) 
mlp_100_100     = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=42) 

mlp_default.fit(imputed_X_train, y_train)
mlp_10.fit(imputed_X_train, y_train)
mlp_50_50.fit(imputed_X_train, y_train)
mlp_100_100.fit(imputed_X_train, y_train)

y_pred_default  = mlp_default.predict(imputed_X_test)
y_pred_10       = mlp_10.predict(imputed_X_test)
y_pred_50_50    = mlp_50_50.predict(imputed_X_test)
y_pred_100_100  = mlp_100_100.predict(imputed_X_test)

acc_default = accuracy_score(y_test, y_pred_default, normalize=True)
f1_default  = f1_score(y_test, y_pred_default, average="binary")

acc_10 = accuracy_score(y_test, y_pred_10, normalize=True)
f1_10  = f1_score(y_test, y_pred_10, average="binary")

acc_50_50 = accuracy_score(y_test, y_pred_50_50, normalize=True)
f1_50_50  = f1_score(y_test, y_pred_50_50, average="binary")

acc_100_100 = accuracy_score(y_test, y_pred_100_100, normalize=True)
f1_100_100 = f1_score(y_test, y_pred_100_100, average="binary")

mlp_end = time.time()


# Open a file to write scores
with open("scores.txt", "w") as file:
    # Write results for KNN
    file.write("KNN Results:\n")
    knn_results = [
        ("Default (n=5)", acc_default, f1_default, knn_end - knn_start),
        ("Small n (n=1)", acc_small_n, f1_small_n, knn_end - knn_start),
        ("Big n (n=10)", acc_big_n, f1_big_n, knn_end - knn_start),
        ("Biggest n (n=50)", acc_biggest_n, f1_biggest_n, knn_end - knn_start)
    ]
    for name, acc, f1, exec_time in knn_results:
        file.write(f"  {name} - Accuracy: {round(acc, 2)}, F1 Score: {round(f1, 2)}, Execution Time: {round(exec_time, 2)} seconds\n")
    file.write("\n")

    # Write results for Random Forest
    file.write("Random Forest Results:\n")
    rf_results = [
        ("Default (n_estimators=100)", acc_default, f1_default),
        ("n_estimators=10", acc_10, f1_10),
        ("n_estimators=50", acc_50, f1_50),
        ("n_estimators=1000", acc_1000, f1_1000)
    ]
    for name, acc, f1 in rf_results:
        file.write(f"  {name} - Accuracy: {round(acc, 2)}, F1 Score: {round(f1, 2)}\n")
    file.write(f"  Execution Time: {round(random_forest_end - knn_end, 2)} seconds\n\n")

    # Write results for MLP
    file.write("MLP Results:\n")
    mlp_results = [
        ("Default (hidden_layer_sizes=(100,))", acc_default, f1_default),
        ("hidden_layer_sizes=(10,)", acc_10, f1_10),
        ("hidden_layer_sizes=(50, 50)", acc_50_50, f1_50_50),
        ("hidden_layer_sizes=(100, 100)", acc_100_100, f1_100_100)
    ]
    for name, acc, f1 in mlp_results:
        file.write(f"  {name} - Accuracy: {round(acc, 2)}, F1 Score: {round(f1, 2)}\n")
    file.write(f"  Execution Time: {round(mlp_end - random_forest_end, 2)} seconds\n\n")

