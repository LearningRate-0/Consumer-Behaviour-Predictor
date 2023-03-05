import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import squarify

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

def swarm_plot(data_dict,x_label,y_label,title,file_name):
    swarm_df = pd.DataFrame.from_dict(data_dict, orient='index').T

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10)) 
    sns.swarmplot(data=swarm_df, ax=ax,size=5)

    # set the plot labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # show the plot
    plt.savefig(file_name, dpi=300) 
    plt.close()

def treemap_plot(total,file_name,Title):
    fig, ax = plt.subplots(figsize=(20, 10))  
    squarify.plot(sizes=total.values(), label=total.keys(), alpha=.8 )
    plt.title(Title)
    plt.axis('off')
    plt.savefig(file_name, dpi=300) 
    plt.close()


# Creates a graph and return dictionary of {'graph_name':'image_source'}
def generate_graph(predictions,file_name):

    cat1 = predictions.groupby('Product_Category_1')['Purchase'].sum().to_dict()
    cat2 = predictions.groupby('Product_Category_2')['Purchase'].sum().to_dict()
    cat3 = predictions.groupby('Product_Category_3')['Purchase'].sum().to_dict()
    total = {}
    for category in range(1,21):
        total[category] = cat1.get(category,0) + cat2.get(category,0) + cat3.get(category,0)
        
    #take top 10 categories by values
    total = dict(sorted(total.items(), key=lambda item: item[1],reverse=True)[:10])
    total = {k: v for k, v in sorted(total.items(), key=lambda item: item[1],reverse=True)}
    
    #make a tree map
    city_dict = {'A':[],'B':[],'C':[]}
    age_dict = {}
    Stay_In_Current_City_Years_dict = {}
    gender_dict = {'M':[],'F':[]}
    marital_status_dict = {0:[],1:[]}

    for ind in predictions.index:
        age = predictions['Age'][ind]
        city = predictions['City_Category'][ind]
        gender = predictions['Gender'][ind]
        marital_status = predictions['Marital_Status'][ind]
        Stay_In_Current_City_Years = predictions['Stay_In_Current_City_Years'][ind]
        city_dict[city].append(predictions['Purchase'][ind])
        marital_status_dict[marital_status].append(predictions['Purchase'][ind]) 
        gender_dict[gender].append(predictions['Purchase'][ind])
        
        if age in age_dict:
            age_dict[age].append(predictions['Purchase'][ind])
        else:
            age_dict[age] = [predictions['Purchase'][ind]]
        if Stay_In_Current_City_Years in Stay_In_Current_City_Years_dict:
            Stay_In_Current_City_Years_dict[Stay_In_Current_City_Years].append(predictions['Purchase'][ind])
        else:
            Stay_In_Current_City_Years_dict[Stay_In_Current_City_Years] = [predictions['Purchase'][ind]]

    for key in age_dict:
        age_dict[key].sort()
        age_dict[key] = age_dict[key][::250]

    for key in Stay_In_Current_City_Years_dict:
        Stay_In_Current_City_Years_dict[key].sort()
        Stay_In_Current_City_Years_dict[key] = Stay_In_Current_City_Years_dict[key][::250]

    for key in city_dict:
        city_dict[key].sort()
        city_dict[key] = city_dict[key][::250]
    
    for key in marital_status_dict:
        marital_status_dict[key].sort()
        marital_status_dict[key] = marital_status_dict[key][::250]
    
    for key in gender_dict:
        gender_dict[key].sort()
        gender_dict[key] = gender_dict[key][::250]

    
    file_path = {}
    file_path['swarm_age'] = 'temp/swarm'+file_name+'_age.png'
    file_path['swarm_Stay_In_Current_City_Years'] = 'temp/swarm'+file_name+'_Stay_In_Current_City_Years.png'
    file_path['swarm_city'] = 'temp/swarm'+file_name+'_City_Category.png'
    file_path['swarm_marital_status'] = 'temp/swarm'+file_name+'_Marital_Status.png'
    file_path['swarm_gender'] = 'temp/swarm'+file_name+'_Gender.png'
    file_path['tree_map_topcat'] = 'temp/tree_map'+file_name+'_categories.png'

    #make a swarm plot
    swarm_plot(age_dict,'Age','Purchase','Purchase vs Age',file_path['swarm_age'])
    swarm_plot(Stay_In_Current_City_Years_dict,'Stay_In_Current_City_Years','Purchase','Purchase vs Stay_In_Current_City_Years',file_path['swarm_Stay_In_Current_City_Years'])
    swarm_plot(city_dict,'City_Category','Purchase','Purchase vs City_Category',file_path['swarm_city'])
    swarm_plot(marital_status_dict,'Marital_Status','Purchase','Purchase vs Marital_Status',file_path['swarm_marital_status'])
    swarm_plot(gender_dict,'Gender','Purchase','Gender vs Purchase',file_path['swarm_gender'])
    treemap_plot(total,file_path['tree_map_topcat'],"Top 10 Categories")
    
    #make a dict with keys as filename ending of above  and value as file name


    return file_path
    

    

    

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
    

