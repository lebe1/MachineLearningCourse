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
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer

# Define column names for the dataset from the hepatitis.names file
columns = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']

df = pd.read_csv('data/hepatitis/hepatitis.data', sep=',', header=None, names=columns)


# Extract the target variable 'Target' as y
y_hepatitis = df[['Class']]

# Extract all other columns as X (excluding 'Target')
X_hepatitis = df.drop('Class', axis=1)

#Preprocess whole train dataset by replacing ? for None value for imputation reasons
X_hepatitis = X_hepatitis.replace('?', np.nan)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_hepatitis, y_hepatitis, test_size=0.2, shuffle=True, random_state=42)


# Set up the colnames
colnames_numerical = ["AGE", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME"]

colnames_categorical = ['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 
            'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']

# Impute missing values
cat_impute_pipe = make_pipeline(SimpleImputer(strategy='most_frequent', missing_values=np.nan))
num_impute_pipe = make_pipeline(SimpleImputer(strategy='median', missing_values=np.nan))


# Scale and log transform all continuous values
scale_pipe = make_pipeline(StandardScaler())
log_pipe = make_pipeline(PowerTransformer())

# One-hot encode all categories represented by numbers (integer)
categorical_pipe = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown='ignore'))

transformer = ColumnTransformer(
    transformers=[
        ("impute_categories", cat_impute_pipe, colnames_categorical),
        ("impute_numerics", num_impute_pipe, colnames_numerical),
        #("scale", scale_pipe, colnames_numerical),
        #("one_hot_encode", categorical_pipe, colnames_categorical),
    ]
)

transformer_pipe = Pipeline([("preprocess", transformer)])

# Set pipelines
knn_pipe = Pipeline([("knn", KNeighborsClassifier())])
random_forest_pipe = Pipeline([("prep", transformer), ("random_forest", RandomForestClassifier(random_state=1))])
mlp_pipe = Pipeline([("mlp", MLPClassifier(random_state=1))])

# Transform X data sets before inserting to model to prevent error for models not able to handle NaNs
transformer_pipe.fit(X_train, y_train)
transformed_data_X_train = pd.DataFrame(transformer_pipe.transform(X_train))

transformer_pipe.fit(X_test)
transformed_data_X_test = pd.DataFrame(transformer_pipe.transform(X_test))



# Convert target value to numpy array
y_test = y_test.values.flatten()
y_train = y_train.values.ravel()

# Fit and predict knn
knn_pipe.fit(transformed_data_X_train, y_train)
predict_proba = knn_pipe.predict_proba(transformed_data_X_test)
predictions = knn_pipe.predict(transformed_data_X_test)
#print("ROC_AUC_SCORE:", roc_auc_score(y_test, predict_proba, multi_class="ovr"))
print("F1_SCORE_MICRO:", f1_score(y_test, predictions, average='micro'))
print("F1_SCORE_MACRO:", f1_score(y_test, predictions, average='macro'))
print("RECAL_MICRO:", recall_score(y_test, predictions, average='micro'))
print("PRECISION_MICRO:", precision_score(y_test, predictions, average='micro'))
print("RECAL_MACRO:", recall_score(y_test, predictions, average='macro'))
print("PRECISION_MACRO:", precision_score(y_test, predictions, average='macro'))
print("ACCURACY:", accuracy_score(y_test, predictions))
print("KNN not predicting positive value:", set(y_test) - set(predictions))



# Fit and predict random forest
random_forest_pipe.fit(X_train, y_train)
predict_proba = random_forest_pipe.predict_proba(X_test)
predictions = random_forest_pipe.predict(X_test)


#print("ROC_AUC_SCORE:", roc_auc_score(y_test, predict_proba, multi_class="ovr"))
print("F1_SCORE_MICRO:", f1_score(y_test, predictions, average='micro'))
print("F1_SCORE_MACRO:", f1_score(y_test, predictions, average='macro'))
print("RECAL_MICRO:", recall_score(y_test, predictions, average='micro'))
print("PRECISION_MICRO:", precision_score(y_test, predictions, average='micro'))
print("RECAL_MACRO:", recall_score(y_test, predictions, average='macro'))
print("PRECISION_MACRO:", precision_score(y_test, predictions, average='macro'))
print("ACCURACY:", accuracy_score(y_test, predictions))

# Fit and predict MLP
mlp_pipe.fit(transformed_data_X_train, y_train)
predict_proba = mlp_pipe.predict_proba(transformed_data_X_test)
predictions = mlp_pipe.predict(transformed_data_X_test)
#print("ROC_AUC_SCORE:", roc_auc_score(y_test, predict_proba))
print("F1_SCORE_MICRO:", f1_score(y_test, predictions, average='micro'))
print("F1_SCORE_MACRO:", f1_score(y_test, predictions, average='macro'))
print("RECAL_MICRO:", recall_score(y_test, predictions, average='micro'))
print("PRECISION_MICRO:", precision_score(y_test, predictions, average='micro'))
print("RECAL_MACRO:", recall_score(y_test, predictions, average='macro'))
print("PRECISION_MACRO:", precision_score(y_test, predictions, average='macro'))
print("ACCURACY:", accuracy_score(y_test, predictions))
print("KNN not predicting positive value:", set(y_test) - set(predictions))

# TODO Fix imputing on both test and train dataset
# TODO Fix scaling error in pipeline leading to dataset not predicting positives