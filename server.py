from flask import Flask, request, jsonify, send_file
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

app = Flask(__name__)


with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # file = request.files['file']
    file = 'dataset/test.csv'
    df = preprocess(file)

    predictions = model.predict(df)
    pd.DataFrame(predictions).to_csv('predictions.csv')
    return send_file('./predictions.csv')

def preprocess(file):
    data = pd.read_csv(file)
    data.drop(["User_ID", "Product_ID"], axis = 1, inplace = True)
    data["Stay_In_Current_City_Years"].replace({'0':0,
                                         '1':1,
                                         '2':4,
                                         '3':3,
                                         '4+':2},inplace = True)

    data["Gender"].replace({'M':1,'F':0},inplace = True)

    data["Age"].replace({'0-17' :17,
                    '18-25':20,
                    '26-35':30,
                    '36-45':40,
                    '46-50':47,
                    '51-55':52,
                    '55+' : 56},
                    inplace = True)
    
    data_city_categories = pd.get_dummies(data['City_Category'])
    data_city_categories = data_city_categories.add_prefix("City_Category_")
    data = pd.concat([data, data_city_categories], axis=1)
    data = data.drop(['City_Category'], axis = 1)
    data['Product_Category_2'] =data['Product_Category_2'].fillna(0)
    data['Product_Category_3'] =data['Product_Category_3'].fillna(0)
    print(data.shape)
    return data


if __name__ == '__main__':
    
    app.run()
    # print(predict())
