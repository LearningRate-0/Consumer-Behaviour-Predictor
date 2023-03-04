with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess(file):
    data = pd.read_csv(file)
    #data.drop(["User_ID", "Product_ID"], axis = 1, inplace = True)
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

def generate_graph(predictions,file_name):
    # Save predictions to csv file    
    pd.DataFrame(predictions).to_csv('temp/'+file_name+'.csv')

    pass

# input_file is a file object
def run_csv_model(input_file,output_file_name):
    # Preprocess data
    data = preprocess(input_file)
    # Predict
    pass

def run_single_model(data):
    

