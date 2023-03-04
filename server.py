# from flask import Flask, request, jsonify
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import math

# app = Flask(__name__)

# Load the model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# @app.route('/predict', methods=['POST'])
def predict():
    # Get the CSV file from the request
    # file = request.files['file']
    file = 'dataset/testing.csv'
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(file)

    # Make predictions using the model
    predictions = model.predict(df)

    # Return the predictions as JSON
    return predictions

if __name__ == '__main__':
    # app.run()
    print(predict())
