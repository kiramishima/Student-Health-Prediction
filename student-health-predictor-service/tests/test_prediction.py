from pathlib import Path
import json
import os
from ..app import predict_single
# from student-health-predictor-service.app import predict_single
import logging
import pickle as pkl

LOGGER = logging.getLogger(__name__)

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

def read_text(file):
    test_directory = Path(__file__).parent
    LOGGER.info(test_directory)
    with open(test_directory / file, 'rt', encoding='utf-8') as f_in:
        return json.load(f_in)

def test_predict_single():
    heart0 = read_text('sample_low.json')
    result = predict_single(heart0)
    LOGGER.info(result)
    expected_result = [1]
    assert result == expected_result
