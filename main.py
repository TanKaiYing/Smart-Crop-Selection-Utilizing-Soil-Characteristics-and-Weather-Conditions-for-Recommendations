import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with header: 
    st.title('Welcome to Harvest Fields for Crop Prediction Service')
    st.markdown('We specialize in providing excellent agricultural services. We offer various farming information services, including crop prediction. We provide the highest quality services to both small and large-scale farms. At Harvest Fields, we believe in using the latest technologies to ensure the highest quality results. We strive to ensure the long-term success of our clients and our business.')

with dataset:
    st.header('Crop Dataset')
    st.text('We have found our dataset on Kaggle')
    
    crop_data = pd.read_csv('data/crop_data.csv')
    st.write(crop_data.head())
    
    st.subheader('Total samples of soil collected for crop prediction')
    types_of_soil = pd.DataFrame(crop_data['SOIL_TYPE'].value_counts()).head(50)
    st.bar_chart(types_of_soil)
    
with features: 
    st.header('Features used for prediction')
    
    st.markdown('* Amount of Nitrogen needed in the soil')
    st.markdown('* Amount of Phosphorous needed in the soil')
    st.markdown('* Amount of Potassium needed in the soil')
    st.markdown('* Surrounding Temperature')
    st.markdown('* Humidity')
    st.markdown('* Amount of Rainfall')
    st.markdown('* pH value of the soil')
    st.markdown('* State')
    st.markdown('* Types of the soil')
    
with model_training:
    st.header('Time to make your crop prediction!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance change') 

    sel_col, disp_col = st.columns(2)
    
    N_SOIL = sel_col.slider('Amount of Nitrogen needed in the soil', min_value=0, max_value=150, value=20, step=5)
    
    P_SOIL = sel_col.slider('Amount of Phosphorous needed in the soil', min_value=0, max_value=150, value=20, step=5)
    
    K_SOIL = sel_col.slider('Amount of Potassium needed in the soil', min_value=0, max_value=150, value=20, step=5)
    
    TEMPERATURE = sel_col.slider('Surrounding Temperature', min_value=5, max_value=50, value=20, step=1)
    
    HUMIDITY = sel_col.slider('Humidity', min_value=0, max_value=100, value=20, step=5)
    
    RAINFALL = sel_col.slider('Amount of Rainfall', min_value=0, max_value=100, value=20, step=1)
    
    ph = sel_col.slider('pH value of the soil', min_value=1, max_value=9, value=20, step=1)
    
    STATE = sel_col.selectbox('State', options=['Kerala', 'Haryana', 'Himachal Pradesh', 'Uttar Pradesh', 'Assam', 'Maharashtra', 'Manipur', 'Madhya Pradesh', 'Gujarat', 'West Bengal', 'Uttrakhand', 'Meghalaya', 'Odisha', 'Nagaland', 'Andhra Pradesh', 'Punjab', 'Tamil Nadu', 'Tripura', 'Telangana', 'Rajasthan', 'Karnataka', 'Pondicherry', 'Goa', 'Chattisgarh', 'Jammu and Kashmir'
], index = 0)

    SOIL_TYPE = sel_col.selectbox('Type of Soil', options=['Sandy Clay loam', 'Desert soil', 'Sandy loam', 'Alluvial soil', 'Laterite soil', 'Regur soil', 'Sandy soil', 'Inceptisols', 'Black soil', 'Desert soils', 'Mountain soil', 'Loamy soil', 'Red soil', 'Delta alluvium', 'Clayey soils'], index = 0)

#prediction code
if st.button('Predict'):
    makeprediction = crop_data.predict([[N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, RAINFALL, ph, STATE, SOIL_TYPE]])
    output=round(makeprediction[0],2)
    st.success('Field conditions are most suitable for {}'.format(output))

if _name_=='_main_':
    main()
