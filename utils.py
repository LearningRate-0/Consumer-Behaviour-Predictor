with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess(data):
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

# Creates a graph and return dictionary of {'graph_name':'image_source'}
def generate_graph(predictions,file_name):
    # Save predictions to csv file    
    

    pass

# input_file is a file object
def run_csv_model(input_file,output_file_name):
    # Read csv from file
    data = pd.read_csv(file)
    # Preprocess data
    processed_data = preprocess(data)
    # Predict
    output_result= model.predict(processed_data)
    # Append output to data
    data['Purchase']=output_result
    pd.DataFrame(data).to_csv('temp/'+output_file_name+'.csv')

    result=generate_graph(predictions, output_file_name)
    result['output_file']='temp/'+output_file_name+'.csv'
    return result
    
# input is a json
def run_single_model(json_data,output_file_name):
    data_dict=json.loads(json_data)
    data=pd.json_normalize(data_dict)
    # Preprocess data
    processed_data = preprocess(data)
    # Predict
    output_result= model.predict(processed_data)
    # Append output to data
    data['Purchase']=output_result
    pd.DataFrame(data).to_csv('temp/'+output_file_name+'.csv')

    result=generate_graph(predictions, output_file_name)
    result['output_file']='temp/'+output_file_name+'.csv'
    return result
    

