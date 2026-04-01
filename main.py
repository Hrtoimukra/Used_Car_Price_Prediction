import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

with open('model.pkl', 'rb') as f:
    model = pk.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pk.load(f)

st.header("Car Price Prediction ML Model")

cars_data=pd.read_csv("Cardetails.csv")

def get_brand_name(car_name):
    brand_name=car_name.split()[0]
    return brand_name.strip()

cars_data['name']=cars_data['name'].apply(get_brand_name)

name=st.selectbox('Select Car Brand',cars_data['name'].unique())
year=st.slider('Car manufacturing year',1994,2024)
km_driven=st.slider('Number of Kms driven',11,200000)
fuel_type=st.selectbox('Select Fuel Type',cars_data['fuel'].unique())
seller_type=st.selectbox('Selct seller Type',cars_data['seller_type'].unique())
transmission=st.selectbox('select transmission type',cars_data['transmission'].unique())
owner_type=st.selectbox('Select owner type',cars_data['owner'].unique())
mileage=st.slider('Mileage',10,40)
engine=st.slider('Select engine in cc',700,5000)
max_power=st.slider('Max power',0,200)
seats=st.selectbox('Select Number of Seats',[2,3,4,5])

if st.button('Predict Price'):
    input_data = pd.DataFrame([[name,year,km_driven,fuel_type,seller_type,transmission,owner_type,mileage,engine,max_power,seats]],
                              columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])

    categorical_columns = ['name','fuel','seller_type','transmission']
    transformed_input_data=pd.get_dummies(input_data,columns=categorical_columns,drop_first=True)

    for col in model_columns:
        if col not in transformed_input_data.columns:
            transformed_input_data[col] = 0

    st.write(transformed_input_data.columns)

    for col in categorical_columns:
        transformed_colm = col+"_"+str(input_data[col].values[0])
        if transformed_colm in model_columns:
           transformed_input_data[col+"_"+str(input_data[col].values[0])]=1

    owner_map = {
        'Test Drive Car': 0,
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }

    transformed_input_data['owner'] = transformed_input_data['owner'].map(owner_map)

    transformed_input_data['km_driven'] = np.log(transformed_input_data['km_driven'])
    transformed_input_data['engine']    = np.log(transformed_input_data['engine'])
    transformed_input_data['age'] = 2025 - transformed_input_data['year']
    transformed_input_data.drop('year', axis=1, inplace=True)

    transformed_input_data = transformed_input_data[model_columns]

    st.write(transformed_input_data)

    car_price= np.exp(model.predict(transformed_input_data))
    st.markdown("Expected Price of the car is" + str(car_price))







