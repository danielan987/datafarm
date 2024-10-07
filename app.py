
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

# Set Streamlit to wide layout mode and add a title for the app
st.set_page_config(layout="wide")
st.title("🌾 Global Soil Moisture Analysis")

# NASA POWER API base URL
NASA_POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Dictionary to map API parameters to user-friendly labels
parameter_labels = {
    "GWETTOP": "0 - 5 cm",
    "GWETROOT": "0 - 100 cm",
    "GWETPROF": "0 to bedrock"
}

# Reverse dictionary to map labels back to API parameters
label_to_parameter = {v: k for k, v in parameter_labels.items()}

# Display the map at the top
st.write("### Select a Location on the Map 🗺️")

# Create a Folium map centered on the Earth with increased width
m = folium.Map(location=[20, 0], zoom_start=2)  # Center the map at a more general location

# Display the map in Streamlit and capture click data
map_data = st_folium(m, width=1200, height=600, returned_objects=["last_clicked"])

# User selection for the parameter (using user-friendly labels)
parameter_label = st.selectbox("Select a parameter to analyze 📏:", list(parameter_labels.values()))
parameter = label_to_parameter[parameter_label]  # Map the selected label back to the API parameter name

# Function to fetch data from NASA POWER API
def fetch_nasa_power_data(lat, lon, parameter):
    # Set start date as January 1, 1981
    start_date = "19810101"  # Define the start date as January 1, 1981
    current_date = datetime.now().strftime("%Y%m%d")  # Get current date in YYYYMMDD format

    # Construct the URL with dynamic dates
    url = f"{NASA_POWER_API}?parameters={parameter}&community=ag&longitude={lon}&latitude={lat}&start={start_date}&end={current_date}&format=JSON"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if 'properties' in data and 'parameter' in data['properties']:
            return data['properties']['parameter']
        else:
            st.error("No data available for the specified soil-level parameter.")
    else:
        st.error(f"It appears the selected area may not contain soil moisture data. Could you please verify the location or select a different area for analysis?")

# Proceed with analysis only if a location is selected and a parameter is chosen
if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    # Display progress while fetching data
    with st.spinner('Fetching data...'):
        data = fetch_nasa_power_data(lat, lon, parameter)

    if data and parameter in data:
        # Convert data to a DataFrame and handle missing values
        df = pd.DataFrame.from_dict(data).replace(-999, np.nan)
        
        # Convert the index to datetime format (assuming dates are in YYYYMMDD format)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')

        st.success("Data fetched successfully! Performing analysis...")

        fig, ax = plt.subplots(figsize=(20, 8))  # Make the graph wider

        # Plot the selected parameter
        ax.plot(df.index, df[parameter], label=f"{parameter_label} (Soil Moisture)", color="#8B4513")

        # Add horizontal reference lines
        ax.axhline(y=0.2, color='red', linestyle='--', label="Reference Line 0.2")  # Red line at 0.2
        ax.axhline(y=0.6, color='blue', linestyle='--', label="Reference Line 0.6")  # Blue line at 0.6

        # Set the y-axis range from 0 to 1
        ax.set_ylim(0, 1)

        # Let Matplotlib handle the x-axis with default formatting
        fig.autofmt_xdate()  # Automatically format x-axis labels for better readability

        # Add a grid
        ax.grid(True)

        # Labels and legend
        ax.set_title("Soil Moisture Historical Analysis")
        ax.set_xlabel("Date")
        ax.set_ylabel("Soil Moisture")
        ax.legend()

        # Show the plot
        st.pyplot(fig)
        
        # Prepare data for Prophet
        df_prophet = df[[parameter]].reset_index()
        df_prophet.columns = ['ds', 'y']  # Prophet expects columns "ds" for date and "y" for values

        # Initialize Prophet model without weekly seasonality
        model = Prophet(weekly_seasonality=False, yearly_seasonality=True)
        model.fit(df_prophet)

        # Make future dataframe for predictions (365 days into the future)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        # Filter the forecast data for the last 365 days
        forecast_zoomed = forecast.tail(365)

        # Plot the forecast for the last 365 days
        fig2, ax2 = plt.subplots(figsize=(20, 8))
        ax2.plot(forecast_zoomed['ds'], forecast_zoomed['yhat'], label="Forecast")
        ax2.fill_between(forecast_zoomed['ds'], forecast_zoomed['yhat_lower'], forecast_zoomed['yhat_upper'], color='lightgray', label="Uncertainty Interval")

        # Add horizontal reference lines to the forecast plot
        ax2.axhline(y=0.2, color='red', linestyle='--', label="Reference Line 0.2")
        ax2.axhline(y=0.6, color='blue', linestyle='--', label="Reference Line 0.6")
        
        # Labels
        ax2.set_title("Soil Moisture Prediction")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Soil Moisture")
        ax2.set_ylim(0, 1)  # Set the y-axis range from 0 to 1

        # Add grid
        ax2.grid(True)

        # Show the zoomed plot
        st.pyplot(fig2)

        # Plot the trend component (without future predictions)
        historical_forecast = forecast[forecast['ds'] <= df_prophet['ds'].max()]  # Only keep historical data

        fig3, ax3 = plt.subplots(figsize=(20, 8))
        ax3.plot(historical_forecast['ds'], historical_forecast['trend'], label="Historical Trend")

        # Add horizontal reference lines to the historical trend plot
        ax3.axhline(y=0.2, color='red', linestyle='--', label="Reference Line 0.2")
        ax3.axhline(y=0.6, color='blue', linestyle='--', label="Reference Line 0.6")

        ax3.set_title("Historical Trend Component from Prophet Model")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Soil Moisture")
        ax3.set_ylim(0, 1)  # Set the y-axis range from 0 to 1
        ax3.grid(True)

        # Show the trend plot
        st.pyplot(fig3)

        # Plot the yearly seasonal cycle with a numerical x-axis (day of the year)
        fig4, ax4 = plt.subplots(figsize=(20, 8))

        # Generate a range of days in a year for the x-axis
        days_in_year = pd.DataFrame({'ds': pd.date_range('2022-01-01', periods=365)})
        seasonal_components = model.predict_seasonal_components(days_in_year)
        
        # Plot the yearly seasonality with numerical x-axis
        ax4.plot(range(1, 366), seasonal_components['yearly'], label="Yearly Seasonality")
        ax4.set_title("Soil Moisture's Seasonal Cycles")
        ax4.set_xlabel("Day of Year")
        ax4.set_ylabel("Impact of Seasonality")

        # Add grid
        ax4.grid(True)

        # Show the seasonal cycle plot
        st.pyplot(fig4)

else:
    st.info("Click on the map to select a location for analysis.")
