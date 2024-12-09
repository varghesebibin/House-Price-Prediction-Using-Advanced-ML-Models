# House Price Prediction Using Advanced ML Models

The House Price Prediction Using Advanced ML Models project predicts house prices using data from the Ames Housing dataset combined with secondary macroeconomic datasets such as CPI, Inflation Rate, and GDP. The models include Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, and Gradient Boosting, with advanced feature engineering and hyperparameter tuning to enhance performance.

The project is built with a focus on:
- Analyzing key features impacting house prices
- Incorporating macroeconomic indicators for additional insights.
- Providing an interactive Streamlit app for exploration and predictions.

<p align="center">
  <img src="https://github.com/varghesebibin/House-Price-Prediction-Using-Advanced-ML-Models/blob/main/HPM.png?raw=true" alt="House Price Prediction Visualization" width="600"/>
</p>

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Project](#how-to-run-the-project)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [Visualizations](#visualizations)
- [Notebook Sections](#notebook-sections)

---

## Introduction

The **House Price Prediction Using Advanced ML Models** project focuses on predicting house prices using a variety of machine learning models, including Ridge, Lasso, Elastic Net, Random Forest, and Gradient Boosting. This project incorporates extensive data preprocessing, feature engineering, and hyperparameter tuning.

Additionally, the project integrates three distinct macroeconomic datasets—Consumer Price Index (CPI), Inflation Rate, and Gross Domestic Product (GDP)—to analyze their impact on housing trends.

This repository includes:

- **House Price Prediction**: Exploratory Data Analysis (EDA), advanced visualizations, feature engineering, and machine learning models.
- **Macroeconomic Analysis**: Analysis of US economic indicators and their relationship to housing prices.

---

## Project Structure

The project contains the following key files and directories:

- `application.py`: Streamlit application for interactive exploration and predictions.
- `ames_data.csv`: Primary dataset used for house price prediction.
- `CPI.csv`: Secondary dataset for tracking consumer price index.
- `US Inflation Rate.csv`: Secondary dataset for tracking inflation rate of US economy.
- `US GDP.csv`: Secondary dataset for tracking GDP of US economy.
- `requirements.txt`: Python library dependencies.
- `House Price Prediction Model Final.ipynb`: Jupyter notebook with detailed analysis
- `README.md`: Documentation for the project.

---

## Datasets

This project uses four datasets:

1. **Ames Housing Dataset**:
   - Source: Ames, Iowa housing data.
   - Description: Contains over 80 features describing various attributes of houses, including quality, size, location, and sale price.

2. **Consumer Price Index (CPI)**:
   - Description: Measures changes in the price level of consumer goods and services purchased by households.

3. **Inflation Rate Dataset**:
   - Description: Tracks the rate at which the general level of prices for goods and services is rising.

4. **Gross Domestic Product (GDP)**:
   - Description: Measures the total economic output of a country.

---

## Data Description

### 1. **House Price Prediction Dataset (ames_data.csv)**

This dataset includes 81 features that can be used to predict house prices. Each row represents a house, and the columns represent various attributes of the house. Here are the details:

#### Target Variable
- **SalePrice**: The target variable representing the final price of each home.

#### House Characteristics
- **MSSubClass**: Type of dwelling.
- **MSZoning**: General zoning classification.
- **LotFrontage**: Linear feet of street connected to the property.
- **LotArea**: Lot size in square feet.
- **Street**: Type of road access.
- **Alley**: Type of alley access.
- **LotShape**: General shape of the property.
- **LandContour**: Flatness of the property.
- **Utilities**: Type of utilities available.
- **LotConfig**: Lot configuration.
- **LandSlope**: Slope of the property.
- **Neighborhood**: Physical locations within Ames city limits.
- **Condition1**: Proximity to various conditions (e.g., Railroad).
- **Condition2**: Proximity to second conditions.
- **BldgType**: Type of dwelling (e.g., Single-family, Townhouse).
- **HouseStyle**: Style of dwelling (1-story, 2-story).

#### Building Characteristics
- **OverallQual**: Rates overall material and finish of the house.
- **OverallCond**: Rates overall condition of the house.
- **YearBuilt**: Year the house was built.
- **YearRemodAdd**: Year of most recent remodel.
- **RoofStyle**: Type of roof.
- **RoofMatl**: Roof material.
- **Exterior1st**: Exterior covering on the house.
- **Exterior2nd**: Exterior covering on the house.
- **MasVnrType**: Masonry veneer type.
- **MasVnrArea**: Masonry veneer area in square feet.
- **ExterQual**: Quality of exterior material.
- **ExterCond**: Condition of the material on the exterior.
- **Foundation**: Type of foundation.

#### Basement Features
- **BsmtQual**: Quality of the basement.
- **BsmtCond**: General condition of the basement.
- **BsmtExposure**: Walkout or garden-level basement walls.
- **BsmtFinType1**: Quality of finished area in the basement.
- **BsmtFinSF1**: Type 1 finished square feet in the basement.
- **BsmtFinType2**: Quality of second finished area.
- **BsmtFinSF2**: Type 2 finished square feet.
- **BsmtUnfSF**: Unfinished square feet in the basement.
- **TotalBsmtSF**: Total square feet of basement area.

#### Flooring and Living Area
- **1stFlrSF**: First-floor square feet.
- **2ndFlrSF**: Second-floor square feet.
- **GrLivArea**: Above-grade (ground) living area in square feet.

#### Bathroom and Bedroom Features
- **FullBath**: Full bathrooms above grade.
- **HalfBath**: Half bathrooms above grade.
- **BedroomAbvGr**: Bedrooms above grade.
- **KitchenAbvGr**: Kitchens above grade.
- **TotRmsAbvGrd**: Total rooms above grade.

#### Garage Features
- **GarageType**: Location of the garage.
- **GarageYrBlt**: Year garage was built.
- **GarageFinish**: Interior finish of the garage.
- **GarageCars**: Size of garage in car capacity.
- **GarageArea**: Size of garage in square feet.
- **GarageQual**: Garage quality.
- **GarageCond**: Garage condition.

#### Other Features
- **Fireplaces**: Number of fireplaces.
- **PavedDrive**: Paved driveway.
- **PoolArea**: Pool area in square feet.
- **Fence**: Fence quality.
- **MiscFeature**: Miscellaneous features.
- **SaleType**: Type of sale.
- **SaleCondition**: Condition of sale.

### 2. **Consumer Price Index (CPI) Data**

This dataset provides yearly CPI  for the United States, reflects changes in the average price level of consumer goods and services.

- **DATE**: The date at which data was recorded.
- **CPI**: Measure of the weighted average of prices of a basket of consumer goods.

### 3. **Inflation Rate Data**

This dataset tracks the rate at which prices rise over time.
- **DATE**: Inflation data recording date.
- **Inflation_Rate**: Percentage change in average price levels over time.

### 4. **Gross Domestic Product (GDP) Data**
- **DATE**: GDP data recording date
- **GDP_Value**: Gross Domestic Product in trillions.

## Features

Key features for house price prediction include:

- **OverallQual**: Quality of materials and finish.
- **Log_GrLivArea**: Log-transformed above-grade living area in square feet.
- **GarageCars**: Size of garage in car capacity.
- **GarageArea**: Garage size in square feet.
- **Log_1stFlrSF**: Log-transformed first-floor square feet.
- **FullBath**: Full bathrooms above grade.
- **YearBuilt**: Year the house was built.
- **Log_TotRmsAbvGrd**: Log-transformed total rooms above grade.
- **YearRemodAdd**: Year of most recent remodel.
- **Log_House_Age**: Log-transformed house age.

## Setup and Installation

To run this project locally, follow these steps:

### Prerequisites
Ensure Python 3.12.4 or later is installed.

1. **Clone the repository**:
    ```bash
    git clone https://github.com/varghesebibin/House-Price-Prediction-Using-Advanced-ML-Models.git
    cd House-Price-Prediction-Using-Advanced-ML-Models
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv streamlitenv
    source streamlitenv/bin/activate  # On Windows: streamlitenv\Scripts\activate
    ```

3. **Install the required dependencies**:
    Make sure you have `pip` installed, then install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the dataset**:
    Ensure the `ames_data.csv`, 'CPI.csv', 'US Inflation Rate.csv', 'US GDP.csv' files are present in the root directory. If it’s missing, upload your dataset or use your own version.

## How to Run the Project

To run the **Streamlit** app, follow these steps:

1. Open your terminal (ensure you are in the root directory of the project).
2. Run the Streamlit app:
    ```bash
    streamlit run application.py
    ```

3. **Access the app**: Open your web browser and go to the local URL provided (`https://cmse830midsemesterproject-gxbhpk4wstunehwvrgvjgk.streamlit.app/`).

## Key Insights

- **Feature Importance**: OverallQual, GrLivArea, and GarageCars are the strongest predictors of sale prices.
- **Effect of House Age**:Older houses negatively impact prices, showing a strong inverse correlation.
- **Macroeconomic Indicators**: GDP showed a positive correlation, while CPI and Inflation had weaker relationships.
- **Model Performance**: Random Forest and Gradient Boosting achieved the best accuracy, with MAE ≈ $21,000 and R² ≈ 0.84

## Technologies Used

- **Python 3.12.4**: Core programming language.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib** and **Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning, particularly for Ridge and Lasso regression.
- **Streamlit**: Interactive web app framework for displaying the project.
- **HiPlot**: Visualization tool for interactive exploration of feature relationships.
- **Altair**: Additional interactive visualizations.
- **Jupyter Notebook**: Used for data exploration, model building, and evaluation.

## Visualizations

The following visualizations are available within the project:

1. **Distribution of Sale Prices**: Displays the distribution of house prices in the dataset.
2. **Scatter Plot of GrLivArea vs SalePrice**: Shows how the ground living area affects sale prices.
3. **Scatter Plot of TotalBsmtSF vs SalePrice**: Analyzes how basement area impacts house prices.
4. **Boxplot of OverallQual vs SalePrice**: Compares overall quality with house prices.
5. **Heatmap of Correlations**: Displays correlations between key numerical features.
6. **HiPlot**: An interactive HiPlot for exploring relationships between key features and sale prices.
7. **US Macroeconomic Indicator Plots**: Visualization of trends in **CPI**, **GDP** and **Inflation**

## Notebook Sections

The **Jupyter Notebook** file (`House Price Prediction Model Final`) contains a comprehensive analysis of the data. The key sections are as follows:

### 1. **Data Exploration**
- Detailed exploration of the dataset's structure, including the distribution of key variables like `SalePrice`, `GrLivArea`, and `OverallQual` (house price).
- Analysis of the US economy using key macroeconomic indicators such as the **CPI**, **GDP**, **Inflation**.
- Insights on how these factors influenced the house price.
- Visualization of correlations and relationships between variables using scatter plots, histograms, and boxplots.

### 2. **Exploratory Data Analysis (EDA)**
- Visualizations to explore the impact of different features on house prices.
- Identifying outliers and multicollinearity between features.

### 3. **Visualization**
- Use of various plotting techniques to visualize feature relationships.
- Heatmaps, pair plots, and HiPlot were used to display multivariate relationships.

### 4. **Imputation of Missing Values**
- Handling missing values using various imputation techniques, including KNN and iterative imputation methods

### 5. **Feature Engineering**
- Transformation of raw features into more useful variables for modeling.
- Creating interaction terms and handling categorical variables.
- Log transformation for positively skewed features
- Encoding of categorical columns

### 6. **Model Building and Evaluation**
- Training idge, Lasso, Elastic Net, Random Forest, and Gradient Boostig models
- Hyperparameter tuning using cross-validation to optimize the models using GridSearchCV
- Model evaluation using performance metrics such as Mean Absolute Error (MAE) and R².


