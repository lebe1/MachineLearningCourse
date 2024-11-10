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
        #("log_transform", log_pipe, colnames_continuous[13]),
        ("one_hot_encode", categorical_pipe, colnames_categorical),
    ]
)


knn_pipe = Pipeline([("prep", transformer), ("knn", KNeighborsClassifier())])
random_forest_pipe = Pipeline([("prep", transformer), ("random_forest", RandomForestClassifier(random_state=1))])
mlp_pipe = Pipeline([("prep", transformer), ("mlp", MLPClassifier(random_state=1))])


# Encode the target
le = LabelEncoder()
y_train = le.fit_transform(y_train.values.ravel())
y_test = le.transform(y_test.values.ravel())


#-------------------------------------------------------------------------------
# Exploration part for transforming numerical data

# transforming_pipeline = Pipeline([("prep", transformer)])
# transforming_pipeline.fit(X_train, y_train)

# apply the pipeline to the training and test data
# X_train_ = transforming_pipeline.transform(X_train)

# get pipeline feature names
# print(transforming_pipeline.get_feature_names_out())

# fig = plt.figure(figsize=(30,30))
# titles = transforming_pipeline.get_feature_names_out()

# # Take length of columns for count i.e. shape[1]
# for count in range(X_train_.shape[1]):

#     plt.subplot(6, 6, count+1)
#     # plt.title(titles[count])

#     plt.hist(X_train_[count])

## Since setting the title makes the plot less understandable, 
## simply print the titles to connect each plot 
# print(titles)

## The conclusion of investigating the subplots is, that a log-transformer is not
## a good choice afterwards since several distributions have a normal distribution
## after the StandardScaler and after the log transformation it is not normally
## distributed anymore

# Also the x-axes value do not improve after log transformation

## Only the 14. plot showed a typical improvement of the log transformation
## print(colnames_continuous[13)
## which is the column 'inflation rate' and therefore stays as the only column to be log transformed
## TODO Since log-transforming results in an not well-known error, this improvement is post-poned

# plt.show()

#-------------------------------------------------------------------------------


# Fit and predict knn
knn_pipe.fit(X_train, y_train)
predict_proba = knn_pipe.predict_proba(X_test)
predictions = knn_pipe.predict(X_test)
print("ROC_AUC_SCORE:", roc_auc_score(y_test, predict_proba, multi_class="ovr"))
print("F1_SCORE_MICRO:", f1_score(y_test, predictions, average='micro'))
print("F1_SCORE_MACRO:", f1_score(y_test, predictions, average='macro'))
print("RECAL_MICRO:", recall_score(y_test, predictions, average='micro'))
print("PRECISION_MICRO:", precision_score(y_test, predictions, average='micro'))
print("RECAL_MACRO:", recall_score(y_test, predictions, average='macro'))
print("PRECISION_MACRO:", precision_score(y_test, predictions, average='macro'))
print("ACCURACY:", accuracy_score(y_test, predictions))

random_forest_pipe.fit(X_train, y_train)
predict_proba = random_forest_pipe.predict_proba(X_test)
predictions = random_forest_pipe.predict(X_test)
print("ROC_AUC_SCORE:", roc_auc_score(y_test, predict_proba, multi_class="ovr"))
print("F1_SCORE_MICRO:", f1_score(y_test, predictions, average='micro'))
print("F1_SCORE_MACRO:", f1_score(y_test, predictions, average='macro'))
print("RECAL_MICRO:", recall_score(y_test, predictions, average='micro'))
print("PRECISION_MICRO:", precision_score(y_test, predictions, average='micro'))
print("RECAL_MACRO:", recall_score(y_test, predictions, average='macro'))
print("PRECISION_MACRO:", precision_score(y_test, predictions, average='macro'))
print("ACCURACY:", accuracy_score(y_test, predictions))

mlp_pipe.fit(X_train, y_train)
predict_proba = mlp_pipe.predict_proba(X_test)
predictions = mlp_pipe.predict(X_test)
print("ROC_AUC_SCORE:", roc_auc_score(y_test, predict_proba, multi_class="ovr"))
print("F1_SCORE_MICRO:", f1_score(y_test, predictions, average='micro'))
print("F1_SCORE_MACRO:", f1_score(y_test, predictions, average='macro'))
print("RECAL_MICRO:", recall_score(y_test, predictions, average='micro'))
print("PRECISION_MICRO:", precision_score(y_test, predictions, average='micro'))
print("RECAL_MACRO:", recall_score(y_test, predictions, average='macro'))
print("PRECISION_MACRO:", precision_score(y_test, predictions, average='macro'))
print("ACCURACY:", accuracy_score(y_test, predictions))

#TODO Improve and compare models now with GridSearch and other ideas like comparing with log-transformation
#TODO Important read: StandardScaler is sensitive to outliers, and the features may scale differently from each other in the presence of outliers.
#TODO No obvious huge outlier detected in histograms but possible optimization: Check outliers via function, set them nan, impute afterwards
#TODO Simpler: Compare results of StandardScaler vs RobustScaler?
