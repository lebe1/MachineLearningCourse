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

sk = StandardScaler()
scaled_X_train = pd.DataFrame(sk.fit_transform(imputed_X_train))
scaled_X_test = pd.DataFrame(sk.fit_transform(imputed_X_test))

print("After scale",scaled_X_train.isna().sum().sum())

# Convert target value to numpy array
y_test = y_test.values.flatten()
y_train = y_train.values.flatten()

# Set pipelines
knn_model = KNeighborsClassifier(leaf_size=2, n_neighbors=5, weights='uniform')
random_forest_model = RandomForestClassifier(random_state=RANDOM_SEED, criterion='gini', max_features='sqrt', n_estimators=50)
mlp_model = MLPClassifier(random_state=RANDOM_SEED, activation='logistic', hidden_layer_sizes=50, learning_rate_init=0.001, max_iter=200, solver='adam')


# Start measure time point
knn_start = time.time()

# Fit and predict knn
knn_model.fit(scaled_X_train, y_train)
knn_predictions = knn_model.predict(scaled_X_test)
f1_knn = f1_score(y_test, knn_predictions, average='macro')
acc_knn = accuracy_score(y_test, knn_predictions)

knn_end = time.time()

# Fit and predict random forest
random_forest_model.fit(scaled_X_train, y_train)
random_forest_predictions = random_forest_model.predict(scaled_X_test)
f1_rf = f1_score(y_test, random_forest_predictions, average='macro')
acc_rf = accuracy_score(y_test, random_forest_predictions)

random_forest_end = time.time()

# Fit and predict MLP
mlp_model.fit(scaled_X_train, y_train)
mlp_predictions = mlp_model.predict(scaled_X_test)
f1_mlp = f1_score(y_test, mlp_predictions, average='macro')
acc_mlp = accuracy_score(y_test, mlp_predictions)

mlp_end = time.time()


# Open a file to write scores
with open("scores.txt", "w") as file:
    file.write(f"KNN\nF1_SCORE_MACRO: {round(f1_knn,2)}\nACCURACY: {round(acc_knn,2)}\n")
    file.write(f"KNN Execution time in s: {round(knn_end - knn_start,2)}\n\n")
    if set(y_test) - set(knn_predictions):
        file.write(f"KNN not predicting both values. Missing value is: {set(y_test) - set(knn_predictions)}\n\n")
    file.write(f"Random Forest\nF1_SCORE_MACRO: {round(f1_rf,2)}\nACCURACY: {round(acc_rf,2)}\n")
    file.write(f"Random Forest Execution time in s: {round(random_forest_end - knn_end,2)}\n\n")
    if set(y_test) - set(random_forest_predictions):
        file.write(f"Random forest not predicting both values. Missing value is: {set(y_test) - set(random_forest_predictions)}\n\n")
    file.write(f"MLP\nF1_SCORE_MACRO: {round(f1_mlp,2)}\nACCURACY: {round(acc_mlp,2)}\n")
    file.write(f"MLP Execution time in s: {round(mlp_end - random_forest_end,2)}\n\n")
    if set(y_test) - set(mlp_predictions):
        file.write(f"MLP not predicting both values. Missing value is: {set(y_test) - set(mlp_predictions)}\n\n")
