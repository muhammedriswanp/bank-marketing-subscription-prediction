from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ── Load pipeline ──────────────────────────────────────────────────────────────
pipeline = joblib.load('models/random_forest_tuned.pkl')

# ── Expected input fields ──────────────────────────────────────────────────────
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


# ── Health check ───────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'Random Forest (Tuned)'}), 200


# ── Predict endpoint ───────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON body received'}), 400

    # ── Level 1: key validation ────────────────────────────────────────────────
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({
            'error'  : 'Missing required fields',
            'missing': missing
        }), 400

    # ── Level 2: type validation ───────────────────────────────────────────────
    for field in NUMERICAL_FIELDS:
        if not isinstance(data[field], (int, float)):
            return jsonify({
                'error': f"'{field}' must be a number, got {type(data[field]).__name__}"
            }), 400

    # ── Level 3: range validation ──────────────────────────────────────────────
    for field, (min_val, max_val) in FIELD_RANGES.items():
        val = data[field]
        if not (min_val <= val <= max_val):
            return jsonify({
                'error': f"'{field}' must be between {min_val} and {max_val}, got {val}"
            }), 400

    # ── Level 4: categorical validation ───────────────────────────────────────
    for field, valid_values in CATEGORICAL_FIELDS.items():
        if data[field] not in valid_values:
            return jsonify({
                'error'        : f"'{field}' has invalid value '{data[field]}'",
                'valid_values' : valid_values
            }), 400

    # ── Predict ────────────────────────────────────────────────────────────────
    try:
        input_df = pd.DataFrame([data])
        prediction      = pipeline.predict(input_df)[0]
        probability     = pipeline.predict_proba(input_df)[0]

        return jsonify({
            'prediction'        : int(prediction),
            'outcome'           : 'Will Subscribe' if prediction == 1 else 'Will Not Subscribe',
            'probability_yes'   : round(float(probability[1]), 4),
            'probability_no'    : round(float(probability[0]), 4),
            'model'             : 'Random Forest (Tuned)'
        }), 200

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=5000)
