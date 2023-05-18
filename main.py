import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



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
    st.text('We specialize in providing excellent agricultural services. We offer various farming information services, including crop prediction. We provide the highest quality services to both small and large-scale farms. At Harvest Fields, we believe in using the latest technologies to ensure the highest quality results. We strive to ensure the long-term success of our clients and our business.')

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
    
    st.markdown('* **First feature:** Temperature')
    st.markdown('* **Second feature:** pH Value of Soil')
    st.markdown('* **Third feature:** Types of Soil')
    
with model_training:
    st.header('Time to make your crop prediction!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance change')

    sel_col, disp_col = st.columns(2)
    
    TEMPERATURE = sel_col.slider('What should be the Temperature of the model?', min_value=6, max_value=50, value=20, step=1)

    ph = sel_col.selectbox('How is the ph-value for plant?', options=[1,2,3,4,5,6,7,8,9], index = 0)
    
    sel_col.text('Here is list of features in ours data:')
    sel_col.write(crop_data.columns)
    
    input_feature = sel_col.text_input('Which feature should be used as the input feature?','K_SOIL')
    
    regr = RandomForestRegressor()
    
    X = crop_data[[input_feature]]
    y = crop_data['K_SOIL']
    
    regr.fit(X, y)
    prediction = regr.predict(X)
    
    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))
    
    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))
    
    disp_col.subheader('R-squared score of the model is:')
    disp_col.write(r2_score(y, prediction))