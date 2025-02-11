import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import random

# Load trained model
with open("congestion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Generate sample congestion data for graph
def generate_congestion_data():
    data = []
    for i in range(10):  # Generate 10 time points
        packet_rate = random.randint(50, 800)
        throughput = random.randint(10000, 400000)
        length = random.randint(60, 150)
        congestion = model.predict(np.array([[packet_rate, throughput, length]]))[0]
        data.append({
            "time": f"Time {i+1}",
            "packet_rate": packet_rate,
            "throughput": throughput,
            "length": length,
            "congestion": congestion
        })
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    packet_rate = float(request.form['packet_rate'])
    throughput = float(request.form['throughput'])
    length = float(request.form['length'])
    
    # Make prediction
    input_data = np.array([[packet_rate, throughput, length]])
    prediction = model.predict(input_data)
    
    # Return result
    result = "Congestion Detected 🚨" if prediction[0] == 1 else "No Congestion ✅"
    return render_template('index.html', prediction=result)

@app.route('/congestion-data')
def congestion_data():
    return jsonify(generate_congestion_data())

if __name__ == '__main__':
    app.run(port=5000, debug=True)
