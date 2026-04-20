import streamlit as st
import pickle
import numpy as np
import pandas as pd # Import pandas

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type_name = st.selectbox('Type',df['TypeName'].unique()) # renamed 'type' to 'type_name' to avoid confusion

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu Brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    # Pre-processing
    ts_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    
    # Avoid division by zero if slider is at 0
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # CREATE A DATAFRAME instead of a numpy array
    # Ensure column names match exactly what was used during training
    query = pd.DataFrame([[
        company, type_name, ram, weight, ts_val, ips_val, ppi, cpu, hdd, ssd, gpu, os
    ]], columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'Cpu Brand', 'HDD', 'SSD', 'Gpu brand', 'OS'])

    # Predict
    try:
        prediction = pipe.predict(query)
        # Using np.exp because the model was likely trained on log-transformed prices
        final_price = int(np.exp(prediction[0]))
        st.title(f"The predicted price is ₹{final_price}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Ensure the column names in the query DataFrame match your training data.")