# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# For model loading
import pickle

# For HiPlot
import hiplot as hip
import streamlit.components.v1 as components

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px

# Set the page layout
st.set_page_config(page_title="House Price Prediction App", layout="wide")



# Load data
@st.cache_data
def load_data():
    ames_data  = pd.read_csv('ames_data.csv')
    cpi_data = pd.read_csv('cpi_data_cleaned.csv')
    inflation_data = pd.read_csv('inflation_data_cleaned.csv')
    gdp_data = pd.read_csv('gdp_data_cleaned.csv')
    return ames_data, cpi_data, inflation_data, gdp_data

# Load the data
ames_data, cpi_data, inflation_data, gdp_data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
sections = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "EDA and Visualization", "Data Cleaning", "Imputation"]
)

# Dataset Overview Section
if sections == 'Dataset Overview':
    st.title("Dataset Overview")

    st.write("""
    In this section, you can explore the datasets used in this project. Below is an overview of each dataset:
    
    1. **Ames Housing Data**: Provides detailed housing information with over 80 features describing various attributes such as quality, size, and sale price.
    2. **CPI (Consumer Price Index) Data**: Tracks changes in the price level of a basket of consumer goods and services over time.
    3. **Inflation Rate Data**: Shows the rate of price increase for goods and services, indicating economic inflation trends.
    4. **GDP Data (Gross Domestic Product)**: Measures the total economic output of a country, adjusted for inflation, providing insights into economic growth.
    """)

    # Select dataset to explore
    dataset_choice = st.selectbox(
        "Select Dataset to Explore",
        ["Ames Housing Data", "CPI Data", "Inflation Rate Data", "GDP Data"]
    )

    # Ames Housing Data
    if dataset_choice == "Ames Housing Data":
        st.subheader("Ames Housing Data")
        st.write("### Basic Information")
        st.text(ames_data.info())
        st.write("### Statistical Summary")
        st.write(ames_data.describe())
        st.write("### Missing Values (Percentage)")
        missing_values = ames_data.isnull().sum() / len(ames_data) * 100
        st.write(missing_values[missing_values > 0])

    # CPI Data
    elif dataset_choice == "CPI Data":
        st.subheader("CPI Data")
        st.write("### Basic Information")
        st.text(cpi_data.info())
        st.write("### Statistical Summary")
        st.write(cpi_data.describe())
        st.write("### Missing Values (Percentage)")
        missing_values = cpi_data.isnull().sum() / len(cpi_data) * 100
        st.write(missing_values[missing_values > 0])

    # Inflation Rate Data
    elif dataset_choice == "Inflation Rate Data":
        st.subheader("Inflation Rate Data")
        st.write("### Basic Information")
        st.text(inflation_data.info())
        st.write("### Statistical Summary")
        st.write(inflation_data.describe())
        st.write("### Missing Values (Percentage)")
        missing_values = inflation_data.isnull().sum() / len(inflation_data) * 100
        st.write(missing_values[missing_values > 0])

    # GDP Data
    elif dataset_choice == "GDP Data":
        st.subheader("GDP Data")
        st.write("### Basic Information")
        st.text(gdp_data.info())
        st.write("### Statistical Summary")
        st.write(gdp_data.describe())
        st.write("### Missing Values (Percentage)")
        missing_values = gdp_data.isnull().sum() / len(gdp_data) * 100
        st.write(missing_values[missing_values > 0])

    # Download cleaned datasets
    st.write("### Download Datasets")
    st.download_button(
        label="Download Ames Housing Data",
        data=ames_data.to_csv(index=False).encode('utf-8'),
        file_name='ames_data_cleaned.csv',
        mime='text/csv'
    )
    st.download_button(
        label="Download CPI Data",
        data=cpi_data.to_csv(index=False).encode('utf-8'),
        file_name='cpi_data.csv',
        mime='text/csv'
    )
    st.download_button(
        label="Download Inflation Data",
        data=inflation_data.to_csv(index=False).encode('utf-8'),
        file_name='inflation_data.csv',
        mime='text/csv'
    )
    st.download_button(
        label="Download GDP Data",
        data=gdp_data.to_csv(index=False).encode('utf-8'),
        file_name='gdp_data.csv',
        mime='text/csv'
    )
