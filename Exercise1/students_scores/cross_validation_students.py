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
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold


# Set random seed
RANDOM_SEED = 12

# Load dataset
df = pd.read_csv('../data/students_data.csv', delimiter=';')

# Extract the target variable 'Target' as y
y_student = df[['Target']]

# Extract all other columns as X (excluding 'Target')
X_student = df.drop('Target', axis=1)

# train/test split
#X_train, X_test, y_train, y_test = train_test_split(X_student, y_student, test_size=0.2, random_state=RANDOM_SEED)

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
#y_train = le.fit_transform(y_train.values.ravel())
#y_test = le.fit_transform(y_test.values.ravel())
y_student = le.fit_transform(y_student.values.ravel())


# Start measure time point
knn_start = time.time()

# Fit and predict knn
# We take 4 splits to divide the whole dataset into 5 splits with 0.2 percent for the test set
cv = KFold(n_splits=(4))
scores = cross_val_score(knn_pipe, X_student, y_student, cv = cv, scoring='accuracy')
predictions = cross_val_predict(knn_pipe, X_student, y_student, cv=cv)
# knn_pipe.fit(X_train, y_train)
# predictions = knn_pipe.predict(X_test)

f1_knn = f1_score(y_student, predictions, average='macro')
acc_knn = accuracy_score(y_student, predictions)

knn_end = time.time()


# Fit and predict random forest
# random_forest_pipe.fit(X_train, y_train)
# predictions = random_forest_pipe.predict(X_test)
scores = cross_val_score(random_forest_pipe, X_student, y_student, cv = cv, scoring='accuracy')
predictions = cross_val_predict(random_forest_pipe, X_student, y_student, cv=cv)
f1_rf = f1_score(y_student, predictions, average='macro')
acc_rf = accuracy_score(y_student, predictions)

random_forest_end = time.time()


# Fit and predict MLP
scores = cross_val_score(mlp_pipe, X_student, y_student, cv = cv, scoring='accuracy')
predictions = cross_val_predict(mlp_pipe, X_student, y_student, cv=cv)
# mlp_pipe.fit(X_train, y_train)
# predictions = mlp_pipe.predict(X_test)
f1_mlp = f1_score(y_student, predictions, average='macro')
acc_mlp = accuracy_score(y_student, predictions)

mlp_end = time.time()

# Open a file to write scores
with open("scores.txt", "w") as file:
    file.write(f"KNN\nF1_SCORE_MACRO: {round(f1_knn,2)}\nACCURACY: {round(acc_knn,2)}\n\n")
    file.write(f"KNN Execution time in s: {round(knn_end - knn_start,2)}\n\n")
    file.write(f"Random Forest\nF1_SCORE_MACRO: {round(f1_rf,2)}\nACCURACY: {round(acc_rf,2)}\n\n")
    file.write(f"Random Forest Execution time in s: {round(random_forest_end - knn_end,2)}\n\n")
    file.write(f"MLP\nF1_SCORE_MACRO: {round(f1_mlp,2)}\nACCURACY: {round(acc_mlp,2)}\n\n")
    file.write(f"MLP Execution time in s: {round(mlp_end - random_forest_end,2)}\n\n")


