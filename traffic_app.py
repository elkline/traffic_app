# Import libraries
import streamlit as st
import pandas as pd
import pickle

 # Header
st.title('Traffic Volume Prediction: A Machine Learning App') 
st.image('traffic_image.gif', width = 400)
st.subheader("Utilize our advanced Machine Learning application to predict traffic volume.") 
st.write("Use the following form to get started") 

traffic_df = pd.read_csv('Traffic_Volume.csv') # Original data to create ML model

# Extracted month, day, hour data
traffic_df['date_time_column'] = pd.to_datetime(traffic_df['date_time'])
traffic_df['month'] = traffic_df['date_time_column'].dt.month_name()
traffic_df['day'] = traffic_df['date_time_column'].dt.day_name()
traffic_df['hour'] = traffic_df['date_time_column'].dt.hour
traffic_df = traffic_df.drop(columns=['date_time_column', 'date_time', 'weather_description'])

# st.dataframe(traffic_df)
traffic_df['holiday'].fillna('None.', inplace=True)

# Converting dataframe back into CSV
traffic_df.to_csv('Traffic_Volume_adjusted.csv', index=False)

# Creating form for user to select data
holiday = st.selectbox('Choose whether today is a designated holiday or not', options = traffic_df['holiday'].unique())
temp = st.number_input('Average temperature in Kelvin', min_value = 0.1) 
rain_1h = st.number_input('Amount in mm of rain that occured in the hour', min_value = 0.1) 
snow_1h = st.number_input('Amount in mm of snow that occured in the hour', min_value = 0.1) 
clouds_all = st.number_input('Percantage of cloud cover', min_value = 0) 

weather_main = st.selectbox('Choose the current weather', options = traffic_df['weather_main'].unique()) 
month = st.selectbox('Choose month', options = sorted(traffic_df['month'].unique())) 
day = st.selectbox('Choose day of the week', options = sorted(traffic_df['day'].unique())) 
hour = st.selectbox('Choose hour', options = sorted(traffic_df['hour'].unique())) 


user_df = {
    'holiday': [holiday],
    'temp': [temp],
    'rain_1h': [rain_1h],
    'snow_1h': [snow_1h],
    'clouds_all': [clouds_all],
    'weather_main': [weather_main],
    'month': [month],
    'day': [day],
    'hour': [hour]
}

user_df = pd.DataFrame(user_df)
traffic_csv_df = pd.read_csv('Traffic_Volume_adjusted.csv')

combined_df = pd.concat([traffic_csv_df, user_df], axis = 0)

# st.write(traffic_csv_df['holiday'].unique())
cat_var = ['holiday', 'weather_main', 'month', 'day']
features_encoded = pd.get_dummies(traffic_csv_df, columns = cat_var)

features_encoded.head()

# st.dataframe(features_encoded)

selected_model = st.selectbox('Select Machine Learning Model for Prediction', options = ['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost']) 

# Decision Tree
dt_pickle = open('dt_traffic.pickle', 'rb') 
clf_dt = pickle.load(dt_pickle) 
dt_pickle.close() 

# st.write(features_encoded.columns)
features_encoded = features_encoded.drop('traffic_volume', axis = 1)
dt_pred = clf_dt.predict(features_encoded)
# st.dataframe(dt_pred)

# Random Forest
rf_pickle = open('rf_traffic.pickle', 'rb') 
clf_rf = pickle.load(rf_pickle) 
rf_pickle.close() 

rf_pred = clf_rf.predict(features_encoded)

# ADABoost
adaB_pickle = open('adaB_traffic.pickle', 'rb') 
clf_adaB = pickle.load(adaB_pickle) 
adaB_pickle.close() 

adaB_pred = clf_adaB.predict(features_encoded)

# XGBoost
xgB_pickle = open('xgB_traffic.pickle', 'rb') 
clf_xgB = pickle.load(xgB_pickle) 
xgB_pickle.close() 

xgB_pred = clf_xgB.predict(features_encoded)


# Results
st.write("These ML models exhibited the following predictive performance on the test dataset") 
results_df = pd.read_csv('results.csv')

max_r2_index = results_df['R2'].idxmax()
min_r2_index = results_df['R2'].idxmin()

# Styling the dataframe
def style_rows(row):
    color = ''  # Default color
    
    if row.name == max_r2_index:
        color = 'green'
    elif row.name == min_r2_index:
        color = 'orange'
    
    return [f'background-color: {color}'] * len(row)

# Apply the style function to the DataFrame
styled_results_df = results_df.style.apply(style_rows, axis=1)
st.dataframe(styled_results_df)

submit_button = st.button("Submit")

# Check if the button is clicked
if submit_button:

    if selected_model == 'Decision Tree':
        st.write("<span style='color:red'>Decision Tree Traffic Prediction:</span>", round(dt_pred[-1]), unsafe_allow_html=True)
        st.subheader("Plot of Feature Importance") 
        st.image('dt_feature_imp.svg')
    elif selected_model == 'Random Forest':
        st.write("<span style='color:red'>Random Forest Traffic Prediction:</span>", round(dt_pred[-1]), unsafe_allow_html=True)
        st.subheader("Plot of Feature Importance") 
        st.image('rf_feature_imp.svg')
    elif selected_model == 'AdaBoost':
        st.write("<span style='color:red'>AdaBoost Traffic Prediction:</span>", round(dt_pred[-1]), unsafe_allow_html=True)
        st.subheader("Plot of Feature Importance") 
        st.image('ada_feature_imp.svg')
    elif selected_model == 'XGBoost':
        st.write("<span style='color:red'>XGBoost Traffic Prediction:</span>", round(dt_pred[-1]), unsafe_allow_html=True)
        st.subheader("Plot of Feature Importance") 
        st.image('xg_feature_imp.svg')