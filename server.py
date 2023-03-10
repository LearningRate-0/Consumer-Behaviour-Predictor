from flask import Flask, request, jsonify, send_file
import pickle
import time
import random
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
import json
from utils import preprocess, model, run_single_model, run_csv_model

#cors
from flask_cors import CORS
app = Flask(__name__, static_url_path="",static_folder=".")
CORS(app)

@app.route('/predict-single',methods=['POST'])
def predictSingle():
    # Get data from request
    data = request.data
    data_dict=json.loads(data)
    data=pd.json_normalize(data_dict)
    # print(data)
    df = pd.DataFrame.from_dict(data)
    # Generate file_name from current time from system in epoch time and random number
    #output_file_name = str(time.time()) + str(random.randint(0, 1000000))
    # Run model
    output_data=run_single_model(df)

    return jsonify(output_data.to_dict())


@app.route('/predict', methods=['POST'])
def predictCsv():
    input_file = request.files['file']
    # input_file = 'dataset/test.csv'
    # Generate file_name from current time from system in epoch time and random number
    output_file_name = str(time.time()) + str(random.randint(0, 1000000))
    # Run model
    paths=run_csv_model(input_file, output_file_name)
    return jsonify(paths)
    

# This function is used to get data by page_number and file_name
@app.route('/result', methods=['GET'])
def result():
    # Get page_number, file_name from request
    page_number = request.args.get('page_number')
    file_name = request.args.get('file_name')
    # Get data from database
    df= pd.read_csv('temp/'+file_name)
    df = df.iloc[(int(page_number)-1)*10:int(page_number)*10]
    # Convert dataframe to json
    data = df.to_json(orient='records')
    return data

# This function is used to download csv file
@app.route('/download', methods=['GET'])
def download():
    # Get file_name from request
    file_name = request.args.get('file_name')
    # Download file
    return send_file('./temp/'+file_name, as_attachment=True)


if __name__ == '__main__':    
    app.run()
    # print(predict())
