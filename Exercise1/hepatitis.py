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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer

# Set random seed
RANDOM_SEED = 12

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

# TODO: Final task if all other finished, try to find a good workaround to do one-hot encoding on this dataset
# onehot_encoder = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown='ignore'))

# encode = ColumnTransformer(
#     transformers=[
#         ("encode_categories", onehot_encoder, colnames_categorical),
#     ]
# )

# encoded_X_train =  pd.DataFrame(encode.fit_transform(imputed_X_train))
# encoded_X_test = pd.DataFrame(encode.fit_transform(imputed_X_test))

sk = StandardScaler()
scaled_X_train = pd.DataFrame(sk.fit_transform(imputed_X_train))
scaled_X_test = pd.DataFrame(sk.fit_transform(imputed_X_test))

print("After scale",scaled_X_train.isna().sum().sum())

# Convert target value to numpy array
y_test = y_test.values.flatten()
y_train = y_train.values.flatten()

# Set pipelines
knn_model = KNeighborsClassifier()
random_forest_model = RandomForestClassifier(random_state=RANDOM_SEED)
mlp_model = MLPClassifier(random_state=RANDOM_SEED, max_iter=500)

# Fit and predict knn
knn_model.fit(scaled_X_train, y_train)
predictions = knn_model.predict(scaled_X_test)
print("F1_SCORE_MACRO:", f1_score(y_test, predictions, average='macro'))
print("ACCURACY:", accuracy_score(y_test, predictions))
if set(y_test) - set(predictions):
    print("KNN not predicting both values. Missing value is: ", set(y_test) - set(predictions))

# Fit and predict random forest
random_forest_model.fit(scaled_X_train, y_train)
predictions = random_forest_model.predict(scaled_X_test)
print("F1_SCORE_MACRO:", f1_score(y_test, predictions, average='macro'))
print("ACCURACY:", accuracy_score(y_test, predictions))

# Fit and predict MLP
mlp_model.fit(scaled_X_train, y_train)
predictions = mlp_model.predict(scaled_X_test)
print("F1_SCORE_MACRO:", f1_score(y_test, predictions, average='macro'))
print("ACCURACY:", accuracy_score(y_test, predictions))
if set(y_test) - set(predictions):
    print("MLP not predicting both values. Missing value is: ", set(y_test) - set(predictions))