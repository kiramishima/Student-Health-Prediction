import os
from flask import Flask
from flask import request
from flask import jsonify
import pickle as pkl
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

model_name = os.getenv("MODEL_NAME")

# Load model
with open(f"{model_name}.pkl", "rb") as f:
    model = pkl.load(f)

# Load Scaler
with open("standard_scaler.bin", "rb") as f:
    scaler = pkl.load(f)

# Load Label
with open("health_risk_level_encoder.bin", "rb") as f:
    le_target = pkl.load(f)

# Load DV
with open("dv.bin", "rb") as f:
    dv = pkl.load(f)

app = Flask('student-health-risk')

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
    X = X.to_dicts()
    X = dv.transform(X)
    X = csr_matrix(X)
    y_predict = model.predict(X)
    return str(le_target.inverse_transform(y_predict)[0])

def predict_single_proba(patient):
    X = prepare_data(patient)
    X = X.to_dicts()
    X = dv.transform(X)
    X = csr_matrix(X)
    y_pred = model.predict_proba(X)
    return y_pred

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    prediction = predict_single(patient)

    result = {
        'heart_failure_risk': prediction,
    }

    return jsonify(result)  ## send back the data in json format to the user

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696)