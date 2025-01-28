# Import packages
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime

# Layout
st.set_page_config(layout="wide")
st.title("🌾 Global Soil Moisture Analysis")

# NASA POWER API base URL
NASA_POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Map API parameters with labels
parameter_labels = {
    "GWETTOP": "0 (surface) to 5 cm below surface",
    "GWETROOT": "0 (surface) to 100 cm below surface",
    "GWETPROF": "0 (surface) to bedrock"
}
label_to_parameter = {v: k for k, v in parameter_labels.items()}

# Folium map
st.write("### Select a Location on the Map 🗺️")
m = folium.Map(location=[20, 0], zoom_start=2)  
map_data = st_folium(m, width=1200, height=600, returned_objects=["last_clicked"])

# Depth of soil selector  
parameter_label = st.selectbox("Select depth of soil to analyze:", list(parameter_labels.values()))
parameter = label_to_parameter[parameter_label]  

# Function to fetch data from the NASA POWER API
def fetch_nasa_power_data(lat, lon, parameter):
    start_date = "19810101"  
    current_date = datetime.now().strftime("%Y%m%d")  
    url = f"{NASA_POWER_API}?parameters={parameter}&community=ag&longitude={lon}&latitude={lat}&start={start_date}&end={current_date}&format=JSON"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'properties' in data and 'parameter' in data['properties']:
            return data['properties']['parameter']
        else:
            st.error("No data available for the specified soil-level parameter.")
    else:
        st.error(f"It appears the selected location/depth may not contain soil moisture data. Could you please verify or select a different location/depth for analysis?")

# Generate analyses
if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    with st.spinner('Fetching data...'):
        data = fetch_nasa_power_data(lat, lon, parameter)
    if data and parameter in data:
        df = pd.DataFrame.from_dict(data).replace(-999, np.nan)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        st.success("Data fetched successfully! Performing analysis...")
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(df.index, df[parameter], label="Soil Moisture", color="gold")
        ax.axhline(y=0.6, color='blue', linestyle='--', label="Too Much Moisture") 
        ax.axhline(y=0.2, color='red', linestyle='--', label="Too Little Moisture") 
        ax.set_ylim(0, 1)
        fig.autofmt_xdate()  
        ax.grid(True)
        st.title("📅")
        ax.set_title("Historical Analysis")
        ax.set_xlabel("Date")
        ax.set_ylabel("Soil Moisture Level")
        ax.legend()
        st.pyplot(fig)
        df_prophet = df[[parameter]].reset_index()
        df_prophet.columns = ['ds', 'y']  
        model = Prophet(weekly_seasonality=False, yearly_seasonality=True)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        forecast_zoomed = forecast.tail(365)
        fig2, ax2 = plt.subplots(figsize=(20, 8))
        ax2.plot(forecast_zoomed['ds'], forecast_zoomed['yhat'], label="Soil Moisture", color="gold")
        ax2.fill_between(forecast_zoomed['ds'], forecast_zoomed['yhat_lower'], forecast_zoomed['yhat_upper'], color='lightgray')
        ax2.axhline(y=0.6, color='blue', linestyle='--', label="Too Much Moisture")
        ax2.axhline(y=0.2, color='red', linestyle='--', label="Too Little Moisture")
        st.title("🔮")
        ax2.set_title("Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Soil Moisture Level")
        ax2.set_ylim(0, 1)  
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
        historical_forecast = forecast[forecast['ds'] <= df_prophet['ds'].max()]  
        fig3, ax3 = plt.subplots(figsize=(20, 8))
        ax3.plot(historical_forecast['ds'], historical_forecast['trend'], label="Soil Moisture", color="gold")
        ax3.axhline(y=0.6, color='blue', linestyle='--', label="Too Much Moisture")
        ax3.axhline(y=0.2, color='red', linestyle='--', label="Too Little Moisture")
        st.title("📈📉")
        ax3.set_title("Trend")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Soil Moisture Level")
        ax3.set_ylim(0, 1) 
        ax3.grid(True)
        ax3.legend()
        st.pyplot(fig3)
        fig4, ax4 = plt.subplots(figsize=(20, 8))
        days_in_year = pd.DataFrame({'ds': pd.date_range('2022-01-01', periods=365)})
        seasonal_components = model.predict_seasonal_components(days_in_year)
        days_in_year['ds'] = pd.to_datetime(days_in_year['ds'])
        days_in_year['month_day'] = days_in_year['ds'].dt.strftime('%m-%d')
        first_day_of_month = days_in_year[days_in_year['ds'].dt.is_month_start]
        st.title("🌷🌻🍁❄️")
        ax4.plot(days_in_year['ds'], seasonal_components['yearly'], label="Seasonality Impact", color="orange")
        ax4.set_xticks(first_day_of_month['ds'])
        ax4.set_xticklabels(first_day_of_month['ds'].dt.strftime('%b %d'))
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_title("Seasonal Cycle")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Impact on Soil Moisture Level")
        ax4.legend()
        ax4.grid(True)
        st.pyplot(fig4)
else:
    st.info("Click on the map to select a location for analysis.")
