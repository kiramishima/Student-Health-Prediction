#!/usr/bin/env python
# coding: utf-8

import os
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, LabelBinarizer
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, classification_report, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import warnings
warnings.filterwarnings('ignore')


# ### Load Data
df = pl.read_csv('../DATASETS/student_health_data.csv')

# Categorical
categorical_df = df.select(
    pl.col(pl.String())
)
categorical_df

categorical = categorical_df.columns
categorical

# Numerical
numerical_df = df.select(
    pl.col(pl.Int64, pl.Float64)
)
numerical_df

numerical = numerical_df.columns
numerical


# ### Mutual info score
pl.Config.set_tbl_rows(20)
df_target = df['Health_Risk_Level']

def calculate_mi(series):
    return mutual_info_score(series, df_target)

df_mi = pl.DataFrame({
    col: calculate_mi(df[col])
    for col in numerical + categorical
}).transpose(include_header=True)

# ### Preprocessing
# #### Encoders
le_gender = LabelEncoder()
le_pa = LabelEncoder()
le_sq = LabelEncoder()
le_md = LabelEncoder()
le_target = LabelEncoder()
stds = StandardScaler()
dv = DictVectorizer()

# transform
df_encode = df.with_columns(
    Gender = le_gender.fit_transform(df['Gender']),
    Physical_Activity = le_pa.fit_transform(df['Physical_Activity']),
    Sleep_Quality = le_sq.fit_transform(df['Sleep_Quality']),
    Mood = le_md.fit_transform(df['Mood']),
    Health_Risk_Level = le_target.fit_transform(df['Health_Risk_Level'])
)
df_encode
le_gender.classes_, le_md.classes_, le_pa.classes_, le_sq.classes_, le_target.classes_

# #### Scaler
# Apply StandarScaler to numerical
scaled_data = stds.fit_transform(df.select(numerical[2:]).to_numpy())
scaled_data

df_scaled = df_encode.with_columns([
    pl.Series(name, scaled_data[:, i]) for i, name in enumerate(numerical[2:])
])

df_target = df_scaled['Health_Risk_Level']

# #### Mutual Score

# Try MI Score again
df_scaled_mi = pl.DataFrame({
    col: calculate_mi(df_scaled[col])
    for col in numerical[1:] + categorical[:-1]
}).transpose(include_header=True)

df_scaled_mi = df_scaled_mi.rename({"column": "Feature", "column_0": "MI Score"}).sort(by='MI Score', descending=True)
df_scaled_mi


# - **Physiological Data**: Real-time biosensor metrics, including heart rate, blood pressure (systolic and diastolic), and stress levels, collected to gauge physical health.
# - **Psychological Data**: Self-reported stress levels and mood states, providing insight into students' mental and emotional well-being.
# 
# > **Health Risk Level**: A target label indicating low, moderate, or high health risk, derived from combinations of physiological and psychological metrics.

# Drop Columns
df_scaled = df_scaled.drop("Student_ID")
df_scaled


# #### Split data

# Split data
df_train_full, df_test = train_test_split(df_scaled, test_size=0.2, shuffle=True, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)


len(df_scaled), len(df_train), len(df_test), len(df_val)

# Train
x_train = df_train.select(pl.exclude(["Student_ID", "Health_Risk_Level", "Physical_Activity", "Sleep_Quality", "Age", "Mood", "Gender"]))
x_train = x_train.to_dicts()
x_train = dv.fit_transform(x_train)
x_train = csr_matrix(x_train)
# Target
y_train = df_train['Health_Risk_Level']

# Validation
x_val = df_val.select(pl.exclude(["Student_ID", "Health_Risk_Level", "Physical_Activity", "Sleep_Quality", "Age", "Mood", "Gender"]))
x_val = x_val.to_dicts()
x_val = dv.transform(x_val)
x_val = csr_matrix(x_val)
# Target
y_val = df_val['Health_Risk_Level']

# Test
x_test = df_test.select(pl.exclude(["Student_ID", "Health_Risk_Level", "Physical_Activity", "Sleep_Quality", "Age", "Mood", "Gender"]))
x_test = x_test.to_dicts()
x_test = dv.transform(x_test)
x_test = csr_matrix(x_test)
# Target
y_test = df_test['Health_Risk_Level']


# ### Train models

# #### LogisticRegression

lr_model = LogisticRegression(random_state=42, solver='newton-cholesky')
lr_model.fit(x_train, y_train)

# Test predict
y_pred_test = lr_model.predict(x_test)
y_pred_val = lr_model.predict(x_val)

# confusion matrix
cf_matrix_test = confusion_matrix(y_test, y_pred_test)
cf_matrix_val = confusion_matrix(y_val, y_pred_val)

cf_matrix_test, cf_matrix_val

fig, ax = plt.subplots(1, 2, figsize=(12, 8))

classes = le_target.inverse_transform(df_target.unique().to_list())
g = sns.heatmap(cf_matrix_test, cmap='Blues', annot=True, fmt="d", cbar=False, ax=ax[0], square=True, xticklabels=classes, yticklabels=classes)
g.set(title="Confusion Matrix Test Predict")

h = sns.heatmap(cf_matrix_val, cmap='Greens', annot=True, fmt="d", cbar=False, ax=ax[1], square=True, xticklabels=classes, yticklabels=classes)
h.set(title="Confusion Matrix Val Predict")
plt.tight_layout()
plt.show()


print(classification_report(y_test, y_pred_test))


print(classification_report(y_val, y_pred_val))


root_mean_squared_error(y_test, y_pred_test), root_mean_squared_error(y_val, y_pred_val)


# #### RandomForestClassifier

# RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(x_train, y_train)

y2_pred_test = rfc.predict(x_test)
y2_pred_val = rfc.predict(x_val)

print(classification_report(y_test, y2_pred_test))
print(classification_report(y_val, y2_pred_val))


root_mean_squared_error(y_test, y2_pred_test), root_mean_squared_error(y_val, y2_pred_val)


# #### RandomForestRegressor

# RandomForest
rfr = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=42, n_jobs=-1)
rfr.fit(x_train, y_train)


y3_pred_test = rfr.predict(x_test)
y3_pred_val = rfr.predict(x_val)


root_mean_squared_error(y_test, y3_pred_test), root_mean_squared_error(y_val, y3_pred_val)


# #### SVC


svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(x_train, y_train)

y4_pred_test = svm.predict(x_test)
y4_pred_val = svm.predict(x_val)


print(classification_report(y_test, y4_pred_test))
print(classification_report(y_val, y4_pred_val))


# ### Model Scores

models = [lr_model, rfc, rfr, svm]
models_names = ["LogisticRegression", "RandomForestClassifier", "RandomForestRegressor", "SVC"]

# Scores
train_score = [model.score(x_train, y_train) for model in models]
test_score = [model.score(x_test, y_test) for model in models]
val_score = [model.score(x_val, y_val) for model in models]

# Measure model state
rate = []
for train, test, val in zip(train_score, test_score, val_score):
    if train <= 0.65 and test <= 0.65 and val <= 0.65:
        rate.append('bad')
    elif (train > 0.65 and train < 0.80) and (test > 0.65 and test < 0.80) and (val > 0.65 and val < 0.80):
        rate.append('middle')
    elif (train >= 0.80 and test >= 0.80 and val >= 0.80) and (train <= 0.999 and test <= 0.999 and val <= 0.999):
        rate.append('good') 
    else:
        rate.append('overfite')  # Handle cases that don't fit the above

# Create DataFrame
model_score = pl.DataFrame({
    'Model': models_names,
    'Train score': [f'{round(score * 100, 2)}%' for score in train_score],
    'Test score': [f'{round(score * 100, 2)}%' for score in test_score],
    'Val score': [f'{round(score * 100, 2)}%' for score in val_score],
    'Evaluate model': rate
})

# Show result:
model_score


# ### Saving models

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


# Select best models
for row in model_score.filter(pl.col("Evaluate model") == "good").iter_rows():
    print(row[0])


# > We can save three models, if we check the dataframe, RFC model has better results. Only for testing, I will save three models.

dest_path = '../models'

for row in tqdm(model_score.filter(pl.col("Evaluate model") == "good").iter_rows()):
        model_name = row[0]
        idx = models_names.index(model_name)
        model = models[idx]
        dump_pickle(model, os.path.join(dest_path, f"{model_name}.pkl"))


# Save enconder and scaler
dump_pickle(le_gender, os.path.join(dest_path, "gender_encoder.bin"))
dump_pickle(le_md, os.path.join(dest_path, "mood_encoder.bin"))
dump_pickle(le_sq, os.path.join(dest_path, "sleep_quality_encoder.bin"))
dump_pickle(le_pa, os.path.join(dest_path, "phisical_activity_encoder.bin"))
dump_pickle(le_target, os.path.join(dest_path, "health_risk_level_encoder.bin"))
dump_pickle(stds, os.path.join(dest_path, "standard_scaler.bin"))
dump_pickle(dv, os.path.join(dest_path, "dv.bin"))

df_train.select(pl.exclude(["Student_ID", "Health_Risk_Level", "Physical_Activity", "Sleep_Quality", "Age", "Mood", "Gender"])).columns


# ### Load model

import pickle as pkl
import json
# Load model
model_name = "RandomForestClassifier"
with open(f"../models/{model_name}.pkl", "rb") as f:
    model = pkl.load(f)

# Load Scaler
with open("../models/standard_scaler.bin", "rb") as f:
    scaler = pkl.load(f)

# Load Label
with open("../models/health_risk_level_encoder.bin", "rb") as f:
    le_target = pkl.load(f)

# Load DV
with open("../models/dv.bin", "rb") as f:
    dv = pkl.load(f)

model, scaler, le_target, dv


with open('../student-health-predictor-service/test/sample_low.json', 'rt', encoding='utf-8') as f_in:
        data1 = json.load(f_in)

with open('../student-health-predictor-service/test/sample_high.json', 'rt', encoding='utf-8') as f_in:
        data2 = json.load(f_in)

with open('../student-health-predictor-service/test/sample_moderate.json', 'rt', encoding='utf-8') as f_in:
        data3 = json.load(f_in)

data2


# #### Test one element

# Prepare
columns = list(data2.keys())[:-1]
# print(columns)
patient = pl.DataFrame([data2]).drop("Health_Risk_Level")
print(patient)
# Scaler
scaled_data = scaler.transform(patient.to_numpy())
print(scaled_data)


patient = patient.with_columns([
    pl.Series(name, scaled_data[:, i]) for i, name in enumerate(columns)
])
patient


X = patient
X = X.to_dicts()
X = dv.transform(X)
X = csr_matrix(X)
y_pred = model.predict(X)
str(le_target.inverse_transform(y_pred)[0])


# #### Adapt to Functions

def prepare_data(patient) -> pl.DataFrame:
    # Prepare
    columns = list(patient.keys())[:-1]
    patient = pl.DataFrame([patient]).drop("Health_Risk_Level")
    # Scaler
    scaled_data = scaler.transform(patient.to_numpy())

    patient = patient.with_columns([
        pl.Series(name, scaled_data[:, i]) for i, name in enumerate(columns)
    ])

    return patient

def predict_single(patient) -> str:
    X = prepare_data(patient)
    # print(X)
    X = X.to_dicts()
    X = dv.transform(X)
    X = csr_matrix(X)
    y_predict = model.predict(X)
    return str(le_target.inverse_transform(y_predict)[0])

for p in tqdm([data1, data2, data3]):
    result = predict_single(p)
    print(result)

