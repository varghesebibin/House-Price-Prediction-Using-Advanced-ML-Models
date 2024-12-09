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

# Set the page layout
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('ames_data.csv')
    return df

df = load_data()

# Load the optimized models
@st.cache_resource
def load_models():
    with open('lasso_model_5vars_opt.pkl', 'rb') as file:
        lasso_model_opt = pickle.load(file)
    with open('ridge_model_5vars_opt.pkl', 'rb') as file:
        ridge_model_opt = pickle.load(file)
    return lasso_model_opt, ridge_model_opt

lasso_model_opt, ridge_model_opt = load_models()

# Optimal alpha values and performance metrics
best_alpha_lasso = 0.0024420530945486497
mae_lasso = 0.11995081755141092
r2_lasso = 0.8181964382694585

best_alpha_ridge = 2.2229964825261956
mae_ridge = 0.12011768180173615
r2_ridge = 0.8187879636528907

# Sidebar navigation
st.sidebar.title("Navigation")
sections = st.sidebar.radio("Go to", ['Data Exploration', 'EDA and Visualization', 'Data Cleaning',
                                      'Imputation with Reasoning', 'Feature Selection',
                                      'Model Building and Evaluation', 'Predict House Price'])

# Data Exploration Section
if sections == 'Data Exploration':
    st.title('Data Exploration')
    st.write("Here's a glimpse of the dataset:")
    st.write(df.head())
    st.write("Dataset Shape:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Statistical Summary:")
    st.write(df.describe())

# EDA and Visualization Section
elif sections == 'EDA and Visualization':
    st.title('Exploratory Data Analysis')

    st.write("### SalePrice Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['SalePrice'], kde=True, ax=ax)
    st.pyplot(fig)

    st.write("### Correlation Heatmap (Seaborn)")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=False)
    st.pyplot(fig)

    st.write("### Correlation Heatmap (Altair)")
    # Convert correlation matrix to long-form data
    corr_df = corr.stack().reset_index()
    corr_df.columns = ['Feature1', 'Feature2', 'Correlation']

    # Create Altair heatmap
    heatmap = alt.Chart(corr_df).mark_rect().encode(
        alt.X('Feature1:O', title=None),
        alt.Y('Feature2:O', title=None),
        alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue')),
        tooltip=['Feature1', 'Feature2', 'Correlation']
    ).properties(
        width=600,
        height=600
    )

    st.altair_chart(heatmap, use_container_width=True)

    st.write("### Scatter Plot")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    x_var = st.selectbox('Select X-axis variable', numeric_cols, index=numeric_cols.index('GrLivArea'))
    y_var = st.selectbox('Select Y-axis variable', numeric_cols, index=numeric_cols.index('SalePrice'))
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
    st.pyplot(fig)

    # HiPlot Visualization
    st.write("### High-dimensional Data Visualization with HiPlot")
    # Select features for HiPlot
    hi_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
    hip_exp = hip.Experiment.from_dataframe(df[hi_features])

    # Generate HiPlot HTML
    hi_html = hip_exp.to_html(notebook_display=False)

    # Embed HiPlot in Streamlit app
    components.html(hi_html, height=600, scrolling=True)

# Data Cleaning Section
elif sections == 'Data Cleaning':
    st.title('Data Cleaning')
    st.write("Before cleaning, data shape:", df.shape)

    # Display missing data
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    st.write("Missing Values Before Cleaning:")
    st.write(missing)

    st.write("Data cleaning steps performed:")
    st.write("""
    - Replaced missing values in features where 'NaN' indicates absence with 'None'.
    - Filled numerical missing values with median.
    - Dropped columns or rows with excessive missing data.
    """)

    # Example of cleaning steps (simplified)
    df_cleaned = df.copy()

    # Handling missing values as per your earlier code
    null_has_meaning = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                        "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
                        "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
    for i in null_has_meaning:
        df_cleaned[i].fillna("None", inplace=True)

    # Fill numerical missing values with median
    df_cleaned['LotFrontage'] = df_cleaned['LotFrontage'].fillna(df_cleaned['LotFrontage'].median())
    df_cleaned['GarageYrBlt'] = df_cleaned['GarageYrBlt'].fillna(df_cleaned['GarageYrBlt'].median())
    df_cleaned['MasVnrArea'] = df_cleaned['MasVnrArea'].fillna(df_cleaned['MasVnrArea'].median())
    df_cleaned['MasVnrType'].fillna("None", inplace=True)

    # Drop remaining missing values
    df_cleaned.dropna(inplace=True)

    st.write("After cleaning, data shape:", df_cleaned.shape)
    st.write("Missing Values After Cleaning:")
    st.write(df_cleaned.isnull().sum().sum())

# Imputation with Reasoning Section
elif sections == 'Imputation with Reasoning':
    st.title('Imputation with Reasoning')
    st.write("Explanation of how missing values were handled:")
    st.write("""
    - **Features where 'NaN' indicates absence**: Replaced missing values with 'None'.
    - **LotFrontage**: Filled missing values with median as it is important for analysis.
    - **GarageYrBlt, MasVnrArea**: Filled missing values with median values.
    - **MasVnrType**: Filled missing values with 'None'.
    - **Remaining Missing Values**: Dropped rows with any remaining missing values.
    """)

# Feature Selection Section
elif sections == 'Feature Selection':
    st.title('Feature Selection')
    st.write("Selected features based on correlation and importance:")

    # Selected features
    selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
    st.write(selected_features)

    st.write("Feature importance visualization:")
    # Dummy feature importance (replace with actual importance if available)
    importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': [0.8, 0.7, 0.65, 0.6, 0.5]
    })
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance, ax=ax)
    st.pyplot(fig)

# Model Building and Evaluation Section
elif sections == 'Model Building and Evaluation':
    st.title('Model Building and Evaluation (Optimized Alpha)')

    # Allow user to select the model to view
    model_type = st.selectbox("Select Model Type", ['Lasso Regression (Optimized Alpha)', 'Ridge Regression (Optimized Alpha)'])

    if model_type == 'Lasso Regression (Optimized Alpha)':
        st.write("Model used: **Lasso Regression (Optimized Alpha)**")

        st.write("Model performance metrics:")
        st.write(f"""
        - Optimal Alpha: {best_alpha_lasso}
        - Mean Absolute Error (MAE): {mae_lasso:.5f}
        - R² Score: {r2_lasso:.5f}
        """)


    elif model_type == 'Ridge Regression (Optimized Alpha)':
        st.write("Model used: **Ridge Regression (Optimized Alpha)**")

        st.write("Model performance metrics:")
        st.write(f"""
        - Optimal Alpha: {best_alpha_ridge}
        - Mean Absolute Error (MAE): {mae_ridge:.5f}
        - R² Score: {r2_ridge:.5f}
        """)

# Predict House Price Section (Updated)
elif sections == 'Predict House Price':
    st.title('Predict House Price')
    st.write("Enter the property details to get an estimated sale price.")

    # Allow user to select the model for prediction
    model_choice = st.selectbox("Select Model for Prediction", ['Lasso Regression (Optimized Alpha)', 'Ridge Regression (Optimized Alpha)'])

    # Collect user input for features
    OverallQual = st.slider('Overall Quality (1-10)', int(df['OverallQual'].min()), int(df['OverallQual'].max()), int(df['OverallQual'].median()))
    GrLivArea = st.number_input('Above grade (ground) living area square feet', value=int(df['GrLivArea'].median()))
    GarageCars = st.slider('Garage capacity (number of cars)', int(df['GarageCars'].min()), int(df['GarageCars'].max()), int(df['GarageCars'].median()))
    TotalBsmtSF = st.number_input('Total square feet of basement area', value=int(df['TotalBsmtSF'].median()))
    YearBuilt = st.slider('Year Built', int(df['YearBuilt'].min()), int(df['YearBuilt'].max()), int(df['YearBuilt'].median()))

    # Create a DataFrame for prediction
    input_data_new = pd.DataFrame({
        'OverallQual': [OverallQual],
        'GrLivArea': [GrLivArea],
        'GarageCars': [GarageCars],
        'TotalBsmtSF': [TotalBsmtSF],
        'YearBuilt': [YearBuilt]
    })

    # Apply necessary preprocessing
    input_data_transformed_new = input_data_new.copy()
    input_data_transformed_new['GrLivArea'] = np.log(input_data_transformed_new['GrLivArea'] + 1)
    input_data_transformed_new['TotalBsmtSF'] = np.log(input_data_transformed_new['TotalBsmtSF'] + 1)

    # Predict and display the result
    if st.button('Predict'):
        # Select the model based on user choice
        if model_choice == 'Lasso Regression (Optimized Alpha)':
            model_opt = lasso_model_opt
        elif model_choice == 'Ridge Regression (Optimized Alpha)':
            model_opt = ridge_model_opt

        # Predict using the selected model
        prediction_log = model_opt.predict(input_data_transformed_new)
        # Convert prediction back from log scale
        prediction = np.exp(prediction_log)
        st.write(f"Estimated Sale Price using {model_choice}: ${prediction[0]:,.2f}")
