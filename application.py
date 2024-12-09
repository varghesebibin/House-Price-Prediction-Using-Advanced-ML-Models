# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import io  # For capturing .info() output

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Set the page layout
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Function to capture .info() output as a string
def get_info_as_string(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

# Load data
@st.cache_data
def load_data():
    ames_data = pd.read_csv('ames_data.csv')
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

    # Select dataset to explore
    dataset_choice = st.selectbox(
        "Select Dataset to Explore",
        ["Ames Housing Data", "CPI Data", "Inflation Rate Data", "GDP Data"]
    )

    # Ames Housing Data
    if dataset_choice == "Ames Housing Data":
        st.subheader("Ames Housing Data")
        st.write("### Basic Information")
        basic_info = get_info_as_string(ames_data)
        st.text(basic_info)
        st.write("### Statistical Summary")
        st.write(ames_data.describe())
        st.write("### Missing Values (Percentage)")
        missing_values = ames_data.isnull().sum() / len(ames_data) * 100
        st.write(missing_values[missing_values > 0])

    # CPI Data
    elif dataset_choice == "CPI Data":
        st.subheader("CPI Data")
        st.write("### Basic Information")
        basic_info = get_info_as_string(cpi_data)
        st.text(basic_info)
        st.write("### Statistical Summary")
        st.write(cpi_data.describe())
        st.write("### Missing Values (Percentage)")
        missing_values = cpi_data.isnull().sum() / len(cpi_data) * 100
        st.write(missing_values[missing_values > 0])

    # Inflation Rate Data
    elif dataset_choice == "Inflation Rate Data":
        st.subheader("Inflation Rate Data")
        st.write("### Basic Information")
        basic_info = get_info_as_string(inflation_data)
        st.text(basic_info)
        st.write("### Statistical Summary")
        st.write(inflation_data.describe())
        st.write("### Missing Values (Percentage)")
        missing_values = inflation_data.isnull().sum() / len(inflation_data) * 100
        st.write(missing_values[missing_values > 0])

    # GDP Data
    elif dataset_choice == "GDP Data":
        st.subheader("GDP Data")
        st.write("### Basic Information")
        basic_info = get_info_as_string(gdp_data)
        st.text(basic_info)
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
        file_name='cpi_data_cleaned.csv',
        mime='text/csv'
    )
    st.download_button(
        label="Download Inflation Data",
        data=inflation_data.to_csv(index=False).encode('utf-8'),
        file_name='inflation_data_cleaned.csv',
        mime='text/csv'
    )
    st.download_button(
        label="Download GDP Data",
        data=gdp_data.to_csv(index=False).encode('utf-8'),
        file_name='gdp_data_cleaned.csv',
        mime='text/csv'
    )