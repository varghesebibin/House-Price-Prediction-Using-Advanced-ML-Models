# application.py

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
    [
        "Dataset Overview", "EDA and Visualization", "Data Cleaning",
        "Feature Engineering", "Principal Component Analysis", 
        "Model Building and Evaluation", "Predict House Price", 
        "Real World Impact"
    ]
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
        st.write("**Description**: Residential housing data from Ames, Iowa. Contains over 80 features related to house quality, size, location, and sale price.")

        # Display Basic Information in a Table
        st.write("### Basic Information")
        basic_info = pd.DataFrame({
            "Column Name": ames_data.columns,
            "Data Type": ames_data.dtypes,
            "Non-Null Count": ames_data.notnull().sum()
        })
        st.write(basic_info)

        # Display Statistical Summary
        st.write("### Statistical Summary")
        st.write(ames_data.describe())

        # Display Missing Values
        st.write("### Missing Values (Percentage)")
        missing_values = ames_data.isnull().sum() / len(ames_data) * 100
        missing_df = missing_values[missing_values > 0].reset_index()
        missing_df.columns = ["Feature", "Missing Value Percentage"]
        st.write(missing_df)

        # Key Information
        st.write("### Key Information")
        st.write("""
        - **Total observations**: 1,460
        - **Total features**: 81
        - **Numeric features**: 38
        - **Categorical features**: 43
        - **Target Variable**: `SalePrice` (no missing values)
        """)

        # Visualize Missing Data
        st.write("### Missing Data Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=missing_df, x="Missing Value Percentage", y="Feature", ax=ax, color="skyblue")
        ax.set_title("Missing Data Percentage by Feature")
        ax.set_xlabel("Missing Percentage")
        ax.set_ylabel("Feature")
        st.pyplot(fig)

        # Impact of Missing Values on Target Variable
        st.write("### Impact of Missing Values on Target Variable (`SalePrice`)")
        features_na = missing_df["Feature"].tolist()

        # Subplots for Missingness vs Median SalePrice
        n_features = len(features_na)
        n_cols = 3
        n_rows = (n_features // n_cols) + (n_features % n_cols > 0)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
        axes = axes.flatten()

        for i, feature in enumerate(features_na):
            temp_data = ames_data.copy()
            temp_data[feature] = np.where(temp_data[feature].isnull(), 1, 0)
            temp_data.groupby(feature)["SalePrice"].median().plot.bar(ax=axes[i], color='skyblue')
            axes[i].set_title(f"{feature} Missingness")
            axes[i].set_ylabel('Median SalePrice')
            axes[i].set_xlabel('Missing (1) or Not Missing (0)')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig)
        
        # Findings and Recommendations
        st.write("### Findings and Recommendations Based on Missingness")
        st.write("""
        **There is a clear relationship between missing values and the sale price of the house. Therefore, the missing values should be replaced with appropriate and meaningful values.**

        ##### These are the findings based on the missingness in the data:

        - PoolQC (99.52%), MiscFeature (96.30%), Alley (93.77%), and Fence (80.75%) have extremely high proportions of missing values. These features may not provide meaningful information and can likely be dropped.
        - MasVnrType (59.72%) and FireplaceQu (47.26%) have moderate missing values. These require imputation as they could be valuable for the model (e.g., masonry veneer and fireplaces impact house price).
        - Features like LotFrontage (17.73%), GarageYrBlt, GarageCond, and basement-related features have low missing percentages and are important predictors. Imputation methods like KNN, stochastic regression, or group-wise median can be applied to retain these features.

        Columns with more than 80% missing values provide limited information and may not contribute meaningfully to the analysis.
        """)

    # CPI Data
    elif dataset_choice == "CPI Data":
        st.subheader("CPI Data (US Economy)")
        st.write("**Description**: Consumer Price Index (CPI) tracks price changes for goods and services purchased by households, a key indicator of inflation trends.")

        # Display Basic Information in a Table
        st.write("### Basic Information")
        basic_info = pd.DataFrame({
            "Column Name": cpi_data.columns,
            "Data Type": cpi_data.dtypes,
            "Non-Null Count": cpi_data.notnull().sum()
        })
        st.write(basic_info)

        # Display Statistical Summary
        st.write("### Statistical Summary")
        st.write(cpi_data.describe())

        # Display Missing Values
        st.write("### Missing Values (Percentage)")
        missing_values = cpi_data.isnull().sum() / len(cpi_data) * 100
        missing_df = missing_values[missing_values > 0].reset_index()
        missing_df.columns = ["Feature", "Missing Value Percentage"]
        st.write(missing_df)

    # Inflation Rate Data
    elif dataset_choice == "Inflation Rate Data":
        st.subheader("Inflation Rate Data (US Economy)")
        st.write("**Description**: Tracks the annual inflation rate, essential for understanding purchasing power changes.")

        # Display Basic Information in a Table
        st.write("### Basic Information")
        basic_info = pd.DataFrame({
            "Column Name": inflation_data.columns,
            "Data Type": inflation_data.dtypes,
            "Non-Null Count": inflation_data.notnull().sum()
        })
        st.write(basic_info)

        # Display Statistical Summary
        st.write("### Statistical Summary")
        st.write(inflation_data.describe())

        # Display Missing Values
        st.write("### Missing Values (Percentage)")
        missing_values = inflation_data.isnull().sum() / len(inflation_data) * 100
        missing_df = missing_values[missing_values > 0].reset_index()
        missing_df.columns = ["Feature", "Missing Value Percentage"]
        st.write(missing_df)

    # GDP Data
    elif dataset_choice == "GDP Data":
        st.subheader("GDP Data (US Economy)")
        st.write("**Description**: Gross Domestic Product (GDP) measures a country's total economic output, a key economic indicator.")

        # Display Basic Information in a Table
        st.write("### Basic Information")
        basic_info = pd.DataFrame({
            "Column Name": gdp_data.columns,
            "Data Type": gdp_data.dtypes,
            "Non-Null Count": gdp_data.notnull().sum()
        })
        st.write(basic_info)

        # Display Statistical Summary
        st.write("### Statistical Summary")
        st.write(gdp_data.describe())

        # Display Missing Values
        st.write("### Missing Values (Percentage)")
        missing_values = gdp_data.isnull().sum() / len(gdp_data) * 100
        missing_df = missing_values[missing_values > 0].reset_index()
        missing_df.columns = ["Feature", "Missing Value Percentage"]
        st.write(missing_df)

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