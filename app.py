import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load trained model, scaler, and feature columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)  # Columns after pd.get_dummies() during training

@app.route('/')
def home():
    return render_template('index.html')  # Simple form page

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_dict = {
        'holiday': request.form['holiday'],           # string
        'temp': float(request.form['temp']),          # float
        'rain': float(request.form['rain']),
        'snow': float(request.form['snow']),
        'weather': request.form['weather'],           # string
        'year': int(request.form['year']),
        'month': int(request.form['month']),
        'day': int(request.form['day']),
        'hour': int(request.form['hours']),
        'minutes': int(request.form['minutes']),
        'seconds': int(request.form['seconds'])
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply one-hot encoding (same as training)
    input_df = pd.get_dummies(input_df)

    # Align columns (handle missing dummies)
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    return render_template('result.html', prediction_text=f"Estimated Traffic Volume: {int(prediction)} vehicles")

if __name__ == "__main__":
    app.run(debug=True)
