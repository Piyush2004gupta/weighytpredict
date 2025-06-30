import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdatapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        height = float(request.form.get('Height'))
        new_data_scaled = standard_scaler.transform([[height]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
