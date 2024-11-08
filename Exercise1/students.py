import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score



# Load dataset
df = pd.read_csv('data/students_data.csv', delimiter=';')

# Extract the target variable 'Target' as y
y_student = df[['Target']]

# Extract all other columns as X (excluding 'Target')
X_student = df.drop('Target', axis=1)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_student, y_student, test_size=0.2, shuffle=True, random_state=42)


# Set up the colnames
colnames_continuous = df.loc[:, 'Curricular units 1st sem (credited)':'GDP'].columns.values

colnames_continuous = np.append(colnames_continuous, ['Previous qualification (grade)','Admission grade', 'Age at enrollment'])

colnames_categorical_1 = df.loc[:, 'Marital status':'Previous qualification'].columns.values

colnames_categorical_2 = df.loc[:, 'Nacionality':"Father's occupation"].columns.values

colnames_categorical = np.append(colnames_categorical_1, colnames_categorical_2)


# TODO Discuss about step log transforming over every variable
# Scale and log transform all continuous values
scale_pipe = make_pipeline(StandardScaler())
log_pipe = make_pipeline(PowerTransformer())

# One-hot encode all categories represented by numbers (integer)
categorical_pipe = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown='ignore'))

transformer = ColumnTransformer(
    transformers=[
        ("scale", scale_pipe, colnames_continuous),
        ("log_transform", log_pipe, colnames_continuous),
        ("one_hot_encode", categorical_pipe, colnames_categorical),
    ]
)


knn_pipe = Pipeline([("prep", transformer), ("knn", KNeighborsClassifier())])


# Encode the target
le = LabelEncoder()
y_train = le.fit_transform(y_train.values.ravel())
y_test = le.transform(y_test.values.ravel())

# Fit/predict/score
_ = knn_pipe.fit(X_train, y_train)
preds = knn_pipe.predict_proba(X_test)

print(roc_auc_score(y_test, preds, multi_class="ovr"))