import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
#from keras.models import load_model
#import keras.engine.topology as KE
st.header(""
         " Visibility Checker in Airports""")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Visibility</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link disabled" href="#1">Hourly Prediction<span class = "sr-only">(PenguinApp.py)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Graphical View</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)
local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
selected = st.text_input("")
button_clicked = st.button("Search")


st.sidebar.write("# Input Values Here")

temperature = st.sidebar.number_input('Temperature(in Celsius)')

humidity = st.sidebar.number_input('Humidity(%)')

wind_speed = st.sidebar.number_input('Wind Speed(m/s)')

dew_point = st.sidebar.number_input('Dew Point Temperature(in Celsius)')

solar_radiation = st.sidebar.number_input('Solar Radiation (MJ/m2)')

rainfall = st.sidebar.number_input('Rainfall (mm)')

snowfall = st.sidebar.number_input('Snowfall (cm)')

click = st.sidebar.button('Predict')

data = { 'Temperature(°C)' : [temperature]
         , 'Humidity(%)': [humidity]
         , 'Wind speed (m/s)' : [wind_speed]
         , 'Dew point temperature(°C)' : [dew_point]
         , 'Solar Radiation (MJ/m2)' : [solar_radiation]
         , 'Rainfall(mm)' : [rainfall]
         , 'Snowfall (cm)' : [snowfall]}
df = pd.DataFrame(data)
col1, col2, col3= st.columns(3)
col1.metric(label = "Temperature(°C)", value = df['Temperature(°C)'], delta = "2.6")
col2.metric(label = "Wind Speed (m/s)", value = df['Wind speed (m/s)'], delta = "1.2")
col3.metric(label = "Humidity(%)", value = df['Humidity(%)'], delta = "3.4")
col1.metric(label = "Dew point temperature(°C)", value = df['Dew point temperature(°C)'], delta = "2.9")
col2.metric(label = "Solar Radiation (MJ/m2)", value = df['Solar Radiation (MJ/m2)'], delta = "0.5")
col3.metric(label = "Rainfall(mm)", value = df['Rainfall(mm)'], delta = "0.8")
col1.metric(label = "Snowfall (cm)", value = df['Snowfall (cm)'], delta = "0")
data = {'lat':[18.943888], 'lon':[72.835991]}
df = pd.DataFrame(data)

st.map(df)