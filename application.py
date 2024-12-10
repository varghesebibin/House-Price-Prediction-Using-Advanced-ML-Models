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
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import plotly.express as px

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
        "Dataset Overview", "Data Cleaning", "Visualizations",
        "Data Integration & Feature Engineering", "Exploratory Data Analysis", "Principal Component Analysis", 
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
        file_name='ames_data.csv',
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

# Data Cleaning Section
if sections == 'Data Cleaning':
    st.title("Data Cleaning")

    # Dataset choice
    dataset_choice = st.selectbox(
        "Select Dataset for Cleaning",
        ["Ames Housing Data", "CPI Data", "Inflation Rate Data", "GDP Data"]
    )

    if dataset_choice == "Ames Housing Data":
        st.subheader("Ames Housing Data Cleaning")
        st.write("""
        **Description**: This dataset contains information about residential houses in Ames, Iowa.
        Cleaning involves handling missing values, removing duplicates, and ensuring data consistency.
        """)

        # Initial Data Shape
        st.write("### Initial Data Shape")
        st.write(f"Rows: {ames_data.shape[0]}, Columns: {ames_data.shape[1]}")

        # Visualize Missing Data Heatmap
        st.write("### Missing Data Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(ames_data.isnull(), cbar=False, cmap='viridis', ax=ax)
        ax.set_title("Initial Missing Data Heatmap")
        st.pyplot(fig)

        # Drop columns with >80% missing values
        st.write("### Dropping Columns with >80% Missing Values")
        high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
        ames_data_cleaned = ames_data.drop(columns=high_missing_cols)
        st.write(f"Dropped columns: {high_missing_cols}")
        st.write(f"Remaining Columns: {ames_data_cleaned.shape[1]}")

        # Handle Missing Values in Numerical Features using KNN
        st.write("### Handling Missing Values for Numerical Features")
        st.write("""
        **Approach**: K-Nearest Neighbors (KNN) imputation is used as it imputes missing values based on the mean of the nearest neighbors.
        This method ensures that the imputed values are consistent with the distribution of the data.
        """)

        from sklearn.impute import KNNImputer
        numerical_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
        knn_imputer = KNNImputer(n_neighbors=5)
        ames_data_cleaned[numerical_features] = knn_imputer.fit_transform(ames_data_cleaned[numerical_features])

        st.write("Remaining missing values in numerical features:")
        st.write(ames_data_cleaned[numerical_features].isnull().sum())

        # Handle Missing Values in Categorical Features using Iterative Imputer
        st.write("### Handling Missing Values for Categorical Features")
        st.write("""
        **Approach**: Iterative Imputation is used to predict missing categorical values by learning from other features. 
        This is done after one-hot encoding categorical features and then decoding the imputed values back to original categories.
        """)

        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        categorical_features = ['MasVnrType', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                                'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
                                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical']

        # Encode categorical features
        encoded_data = pd.get_dummies(ames_data_cleaned[categorical_features], drop_first=True)
        st.write(f"Categorical features expanded to {encoded_data.shape[1]} columns after one-hot encoding.")

        # Apply Iterative Imputer
        iterative_imputer = IterativeImputer(max_iter=10, random_state=0)
        imputed_data = iterative_imputer.fit_transform(encoded_data)
        imputed_df = pd.DataFrame(imputed_data, columns=encoded_data.columns)

        # Decode imputed data back to categorical format
        for feature in categorical_features:
            feature_encoded_cols = [col for col in encoded_data.columns if feature in col]
            reconstructed_data = imputed_df[feature_encoded_cols].idxmax(axis=1)
            category_mapping = {col: col.split('_')[-1] for col in feature_encoded_cols}
            ames_data_cleaned[feature] = reconstructed_data.map(category_mapping)

        st.write("Categorical features imputed and decoded back to original categories.")

        # Visualize Missing Data Heatmap After Cleaning
        st.write("### Missing Data Heatmap After Cleaning")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(ames_data_cleaned.isnull(), cbar=False, cmap='viridis', ax=ax)
        ax.set_title("Post-Cleaning Missing Data Heatmap")
        st.pyplot(fig)

        # Final Cleaned Data Shape
        st.write("### Final Cleaned Data Shape")
        st.write(f"Rows: {ames_data_cleaned.shape[0]}, Columns: {ames_data_cleaned.shape[1]}")

        # Show Cleaned Data
        st.write("### Cleaned Dataset Preview")
        st.dataframe(ames_data_cleaned.head())

        # Download Cleaned Data
        st.download_button(
            label="Download Cleaned Ames Housing Data",
            data=ames_data_cleaned.to_csv(index=False).encode('utf-8'),
            file_name='ames_data_cleaned.csv',
            mime='text/csv'
        )

    else:
        st.subheader(f"{dataset_choice} Cleaning")
        st.write("**No missing values detected. No cleaning required.**")
        st.write(f"Rows: {cpi_data.shape[0] if dataset_choice == 'CPI Data' else inflation_data.shape[0] if dataset_choice == 'Inflation Rate Data' else gdp_data.shape[0]}")
        st.write(f"Columns: {cpi_data.shape[1] if dataset_choice == 'CPI Data' else inflation_data.shape[1] if dataset_choice == 'Inflation Rate Data' else gdp_data.shape[1]}")

        # Preview Dataset
        st.write("### Dataset Preview")
        st.dataframe(cpi_data.head() if dataset_choice == 'CPI Data' else inflation_data.head() if dataset_choice == 'Inflation Rate Data' else gdp_data.head())

        # Download Dataset
        st.download_button(
            label=f"Download Cleaned {dataset_choice}",
            data=cpi_data.to_csv(index=False).encode('utf-8') if dataset_choice == 'CPI Data' else inflation_data.to_csv(index=False).encode('utf-8') if dataset_choice == 'Inflation Rate Data' else gdp_data.to_csv(index=False).encode('utf-8'),
            file_name=f"{dataset_choice.lower().replace(' ', '_')}_cleaned.csv",
            mime='text/csv'
        )

# Load cleaned data function (reusable and hidden from UI)
@st.cache_data
def load_cleaned_data():
    ames_data_cleaned = pd.read_csv('ames_data_cleaned.csv')
    cpi_data_cleaned = pd.read_csv('cpi_data_cleaned.csv')
    inflation_data_cleaned = pd.read_csv('inflation_data_cleaned.csv')
    gdp_data_cleaned = pd.read_csv('gdp_data_cleaned.csv')
    return ames_data_cleaned, cpi_data_cleaned, inflation_data_cleaned, gdp_data_cleaned

# Load datasets globally (hidden from UI)
ames_data, cpi_data, inflation_data, gdp_data = load_cleaned_data()

# Visualizations Section
if sections == "Visualizations":
    st.title("Visualizations")

    # CPI Visualization
    st.subheader("CPI Trends Over Time")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=cpi_data, x='Year', y='CPI', marker='o', color='green', label='CPI', ax=ax)
    ax.set_title('CPI Trends Over Time')
    ax.set_xlabel('Year')
    ax.set_ylabel('Consumer Price Index (CPI)')
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("""
    The Consumer Price Index (CPI) shows a clear upward trend, indicating inflation over the years. 
    **In the U.S., housing accounts for about 45% of the core CPI basket**, highlighting the relevance of CPI in house price analysis.
    """)

    # Inflation Rate Visualization
    st.subheader("Inflation Rate Trends Over Time")
    fig = px.line(
        inflation_data,
        x='Year',
        y='Inflation_Rate',
        title='Inflation Rates Over Time',
        labels={'Inflation_Rate': 'Inflation Rate (%)', 'Year': 'Year'},
        markers=True
    )
    fig.update_traces(
        line=dict(color='blue', width=2),
        marker=dict(size=6, symbol='circle', color='red')
    )
    fig.update_layout(
        hovermode='x unified',
        title_font=dict(size=22, family='Arial', color='darkblue'),
        xaxis=dict(title='Year', showgrid=True),
        yaxis=dict(title='Inflation Rate (%)', showgrid=True),
        template='plotly_white'
    )
    st.plotly_chart(fig)

    st.markdown("""
    This visualization highlights the high volatility in inflation rates during earlier years, with stabilization post-1950.
    Understanding inflation rates helps in contextualizing house price trends and their real-world implications.
    """)

    # GDP Visualization
    st.subheader("GDP Trends Over Time")
    alt_chart = alt.Chart(gdp_data).mark_line(point=True).encode(
        x=alt.X('Year:Q', title='Year'),
        y=alt.Y('GDP_Value (Trillions):Q', title='GDP (in Trillions)'),
        tooltip=['Year', 'GDP_Value (Trillions)']
    ).properties(
        title='GDP Trends Over Time',
        width=800,
        height=400
    ).interactive()
    st.altair_chart(alt_chart)

    st.markdown("""
    The Gross Domestic Product (GDP) shows consistent growth over decades, with significant acceleration post-1950. 
    Notable slowdowns occurred during the 2008 financial crisis, while a strong recovery is evident post-2020.
    """)

    # Ames Housing Data Visualizations
    st.subheader("Ames Housing Data Visualizations")

    # Distribution of SalePrice
    st.write("### Distribution of Sale Prices")
    fig, ax = plt.subplots()
    sns.histplot(ames_data['SalePrice'], kde=True, ax=ax)
    ax.set_title('Distribution of Sale Prices')
    ax.set_xlabel('Sale Price')
    ax.set_ylabel('Density')
    st.pyplot(fig)

    st.markdown("""
    The distribution plot of house prices shows that the `SalePrice` variable is right-skewed, 
    indicating that most houses are priced lower, but there are a few high-priced houses in the dataset.
    """)

    # Scatter Plot
    st.write("### Scatter Plot of Sale Price vs Other Features")
    numeric_cols = ames_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols.remove('SalePrice')  # Remove SalePrice from the options
    scatter_feature = st.selectbox("Select a feature to compare with SalePrice:", numeric_cols)
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=ames_data, x=scatter_feature, y='SalePrice', ax=ax)
    ax.set_title(f'Scatter Plot of {scatter_feature} vs Sale Price')
    ax.set_xlabel(scatter_feature)
    ax.set_ylabel('Sale Price')
    st.pyplot(fig)

    st.markdown(f"""
    The scatter plot shows the relationship between `{scatter_feature}` and `SalePrice`. 
    This allows for a better understanding of how changes in `{scatter_feature}` might influence house prices.
    """)

    # Boxplot for Overall Quality vs Sale Price
    st.write("### Boxplot of Overall Quality vs Sale Price")
    fig, ax = plt.subplots()
    sns.boxplot(data=ames_data, x='OverallQual', y='SalePrice', ax=ax)
    ax.set_title('Boxplot of Overall Quality vs Sale Price')
    ax.set_xlabel('Overall Quality')
    ax.set_ylabel('Sale Price')
    st.pyplot(fig)

    st.markdown("""
    The boxplot above shows that houses with higher quality ratings tend to have higher sale prices.
    Notably, the median sale price increases consistently with the quality rating (1-10), and there are several outliers, 
    especially in lower quality categories (see OverallQual = 5), where a few houses are priced much higher than the majority of houses with the same quality rating (OverallQual 6 &7).
    """)

# Data Integration & Feature Engineering Section
# Load Cleaned Data
@st.cache_data
def load_cleaned_data():
    ames_data_cleaned = pd.read_csv('ames_data_cleaned.csv')
    cpi_data_cleaned = pd.read_csv('cpi_data_cleaned.csv')
    inflation_data_cleaned = pd.read_csv('inflation_data_cleaned.csv')
    gdp_data_cleaned = pd.read_csv('gdp_data_cleaned.csv')
    return ames_data_cleaned, cpi_data_cleaned, inflation_data_cleaned, gdp_data_cleaned

# Load the cleaned datasets
ames_data, cpi_data_cleaned, inflation_data_cleaned, gdp_data_cleaned = load_cleaned_data()

# Data Integration & Feature Engineering Section
if sections == "Data Integration & Feature Engineering":
    st.title("Data Integration & Feature Engineering")

    # Markdown: Integration Overview
    st.markdown("""
    ### Integration Overview
    The aim is to enhance the Ames Housing dataset by incorporating macroeconomic indicators such as **CPI**, **Inflation Rate**, 
    and **GDP** for the years corresponding to the house sale (`YrSold`). These features help to analyze and adjust house prices 
    in the context of economic trends.
    """)

    # Explanation for Using `YrSold`
    st.markdown("""
    **Why `YrSold`?**
    - The `YrSold` feature directly corresponds to the transaction year when the house price (`SalePrice`) was finalized.
    - This makes it the most relevant feature for linking house price data with macroeconomic indicators like GDP, CPI, and Inflation.
    """)

    # Display Unique Years in YrSold
    unique_years = ames_data['YrSold'].unique()
    st.markdown("#### Unique Years in Ames Housing Data (`YrSold`)")
    st.write(f"Unique Years: {unique_years}")
    st.write(f"Year Range: {ames_data['YrSold'].min()} - {ames_data['YrSold'].max()}")

    # Filter datasets to align with YrSold range
    cpi_filtered = cpi_data_cleaned[(cpi_data_cleaned['Year'] >= 2006) & (cpi_data_cleaned['Year'] <= 2010)]
    inflation_filtered = inflation_data_cleaned[(inflation_data_cleaned['Year'] >= 2006) & (inflation_data_cleaned['Year'] <= 2010)]
    gdp_filtered = gdp_data_cleaned[(gdp_data_cleaned['Year'] >= 2006) & (gdp_data_cleaned['Year'] <= 2010)]

    # Display filtered datasets
    st.markdown("### Filtered Datasets")
    st.markdown("#### Filtered CPI Data")
    st.write(cpi_filtered.head())
    st.markdown("#### Filtered Inflation Data")
    st.write(inflation_filtered.head())
    st.markdown("#### Filtered GDP Data")
    st.write(gdp_filtered.head())

    # Merge datasets with Ames Housing
    st.markdown("### Merging Macroeconomic Indicators with Ames Housing Data")
    ames_data = ames_data.merge(cpi_filtered[['Year', 'CPI']], left_on='YrSold', right_on='Year', how='left')
    ames_data = ames_data.merge(inflation_filtered[['Year', 'Inflation_Rate']], left_on='YrSold', right_on='Year', how='left')
    ames_data = ames_data.merge(gdp_filtered[['Year', 'GDP_Value (Trillions)']], left_on='YrSold', right_on='Year', how='left')

    # Drop redundant year columns
    ames_data = ames_data.drop(columns=['Year_x', 'Year_y', 'Year'])

    st.markdown("#### Columns After Merging")
    st.write(ames_data.columns.tolist())

    # Feature Engineering
    st.markdown("### Feature Engineering")
    st.markdown("#### House Age Feature")
    ames_data['House_Age'] = ames_data['YrSold'] - ames_data['YearBuilt']
    ames_data.loc[ames_data['House_Age'] < 0, 'House_Age'] = 0  # Handle negative ages if any
    st.markdown("- Created a new feature `House_Age` to capture the age of the house at the time of sale.")
    st.write(ames_data[['YrSold', 'YearBuilt', 'House_Age']].head())

    # Inflation Adjusted Price Calculation
    st.markdown("""
    ### Inflation Adjusted Price Calculation
    Using the formula:
    $$ Inflation \\ Adjusted \\ Price = SalePrice \\times \\left( \\frac{Base \\ CPI (2010)}{CPI \\ for \\ YrSold} \\right) $$
    The 2010 CPI is used as the base (100). This calculation adjusts nominal house prices to account for inflation, enabling accurate comparisons across years.
    """)
    base_cpi = cpi_filtered[cpi_filtered['Year'] == 2010]['CPI'].values[0]
    ames_data['Inflation_Adjusted_Price'] = ames_data['SalePrice'] * (base_cpi / ames_data['CPI'])

    # Display Sample Data
    st.markdown("#### Sample of Data After Integration")
    st.write(ames_data[['YrSold', 'YearBuilt', 'House_Age', 'SalePrice', 'Inflation_Adjusted_Price', 'CPI', 'Inflation_Rate', 'GDP_Value (Trillions)']].head())

    # Reasoning Behind Feature Engineering
    st.markdown("""
    ### Reasoning Behind Feature Engineering
    1. **House Age**:
       - Calculated as the difference between `YrSold` and `YearBuilt`.
       - A critical predictor, as newer houses tend to have higher sale prices.
       - Any negative values (possible due to inconsistent data) are set to zero.
    2. **Inflation Adjusted Price**:
       - Accounts for the change in purchasing power over time.
       - Nominal prices are adjusted using CPI to make house prices comparable across years.
    """)

    # Visualize Inflation Adjusted Prices
    st.markdown("#### Visualization: Nominal vs. Inflation Adjusted Prices")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=ames_data, x='SalePrice', y='Inflation_Adjusted_Price', hue='YrSold', palette='viridis', ax=ax)
    ax.set_title('Nominal vs. Inflation Adjusted Prices')
    ax.set_xlabel('Nominal Sale Price')
    ax.set_ylabel('Inflation Adjusted Price')
    st.pyplot(fig)

    st.markdown("""
    The scatter plot illustrates the relationship between nominal sale prices and inflation-adjusted prices.
    For earlier years (2006–2008), inflation-adjusted prices are slightly higher due to lower CPI values relative to 2010.
    """)

    # Columns in Final Dataset
    st.markdown("#### Final Dataset Columns")
    st.write(ames_data.columns.tolist())

    # Download Final Integrated Dataset
    st.markdown("### Download Final Integrated Dataset")
    st.download_button(
        label="Download Integrated Dataset",
        data=ames_data.to_csv(index=False).encode('utf-8'),
        file_name='ames_data_integrated.csv',
        mime='text/csv'
    )

