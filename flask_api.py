# flask_api.py
# Flask REST API for Bank Marketing Subscription Prediction

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ── Load pipeline ──────────────────────────────────────────────────────────────
pipeline = joblib.load('models/random_forest_tuned.pkl')
print(f"[INFO] Loaded model: {type(pipeline.named_steps['classifier']).__name__}")

# ── Field definitions ──────────────────────────────────────────────────────────
REQUIRED_FIELDS = [
    'age', 'job', 'marital', 'education', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'campaign', 'previous',
    'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed', 'previous_contact'
]

NUMERICAL_FIELDS = [
    'age', 'campaign', 'previous', 'emp.var.rate',
    'cons.price.idx', 'cons.conf.idx', 'euribor3m',
    'nr.employed', 'previous_contact'
]

FIELD_RANGES = {
    'age'           : (18, 100),
    'campaign'      : (1, 50),
    'previous'      : (0, 20),
    'emp.var.rate'  : (-5.0, 5.0),
    'cons.price.idx': (90.0, 95.0),
    'cons.conf.idx' : (-55.0, 0.0),
    'euribor3m'     : (0.5, 6.0),
    'nr.employed'   : (4900.0, 5300.0),
    'previous_contact': (0, 1)
}

CATEGORICAL_FIELDS = {
    'job'        : ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                    'management', 'retired', 'self-employed', 'services',
                    'student', 'technician', 'unemployed'],
    'marital'    : ['divorced', 'married', 'single'],
    'education'  : ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
                    'high.school', 'professional.course', 'university.degree'],
    'housing'    : ['yes', 'no'],
    'loan'       : ['yes', 'no'],
    'contact'    : ['cellular', 'telephone'],
    'month'      : ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
    'poutcome'   : ['failure', 'nonexistent', 'success']
}


# ── Input validation ───────────────────────────────────────────────────────────
def validate_input(data):
    # Level 1 — required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            return f"Missing required field: '{field}'"

    # Level 2 — type validation
    for field in NUMERICAL_FIELDS:
        if not isinstance(data[field], (int, float)):
            return f"'{field}' must be a number, got {type(data[field]).__name__}"

    # Level 3 — range validation
    for field, (min_val, max_val) in FIELD_RANGES.items():
        val = data[field]
        if not (min_val <= val <= max_val):
            return f"'{field}' must be between {min_val} and {max_val}, got {val}"

    # Level 4 — categorical validation
    for field, valid_values in CATEGORICAL_FIELDS.items():
        if data[field] not in valid_values:
            return f"'{field}' has invalid value '{data[field]}'. Valid: {valid_values}"

    return None  # None means no error


# ── Health check ───────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model' : 'Random Forest (Tuned)'
    }), 200


# ── Predict endpoint ───────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({'error': 'Request body must be JSON'}), 400

    # run validation
    error = validate_input(data)
    if error:
        return jsonify({'error': error}), 400

    # predict
    try:
        input_df   = pd.DataFrame([data])
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]

        return jsonify({
            'prediction'     : int(prediction),
            'outcome'        : 'Will Subscribe' if prediction == 1 else 'Will Not Subscribe',
            'probability_yes': round(float(probability[1]), 4),
            'probability_no' : round(float(probability[0]), 4),
            'model'          : 'Random Forest (Tuned)'
        }), 200

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)