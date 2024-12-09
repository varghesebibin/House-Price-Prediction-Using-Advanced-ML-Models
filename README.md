# House Price Prediction Using Advanced ML Models

<p align="center">
  <img src="https://github.com/varghesebibin/house-price-prediction/blob/main/images/house_price_visualization.png" alt="House Price Prediction Visualization" width="600"/>
</p>

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Data Description](#data-description)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Project](#how-to-run-the-project)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [Visualizations](#visualizations)
- [Notebook Sections](#notebook-sections)

---

## Introduction

The **House Price Prediction Using Advanced ML Models** project analyzes and predicts house prices using data from the Ames Housing dataset. In addition to house-specific features, this project incorporates three macroeconomic datasets: **CPI**, **Inflation Rate**, and **GDP**, to evaluate external factors influencing housing trends. The models include **Linear Regression**, **Ridge**, **Lasso**, **Elastic Net**, **Random Forest**, and **Gradient Boosting**.

The project follows the **ABCDE Framework**:

1. **Ask**: What features most influence house prices, and how do macroeconomic indicators relate to housing trends?
2. **Breakdown**: Use advanced ML models and interpret feature contributions to predictions.
3. **Create**: Perform feature engineering, hyperparameter tuning, and log-transformed regression models.
4. **Deliver**: Develop a user-friendly **Streamlit app** for interactive exploration and predictions.
5. **Evaluate**: Measure model performance using metrics like **MAE**, **RMSE**, and **R²**.

---

## Project Structure

This repository contains the following files and directories:

- `application.py`: Streamlit application for deploying the project.
- `ames_data.csv`: Primary dataset for house price predictions.
- `requirements.txt`: Python dependencies required to run the project.
- `House_Price_Prediction.ipynb`: Jupyter notebook for exploratory analysis, feature engineering, and model evaluation.
- `README.md`: Documentation for the repository.

---

## Datasets

The project uses the following datasets:

1. **Ames Housing Dataset** (`ames_data.csv`): 
   - Source: [Ames Housing Dataset](https://www.kaggle.com/prevek18/ames-housing-data)
   - Includes 81 features describing houses sold in Ames, Iowa.

2. **Consumer Price Index (CPI)**:
   - Reflects changes in the average price level of consumer goods and services.

3. **Inflation Rate Dataset**:
   - Tracks the rate at which prices rise over time.

4. **Gross Domestic Product (GDP)**:
   - Measures the overall economic activity.

The datasets were cleaned, imputed, and merged for a cohesive analysis.

---

## Data Description

### Target Variable
- **Log_Inflation_Adjusted_Price**: Log-transformed sale prices for improved model performance.

### Key Features for Prediction:
1. **OverallQual**: Overall material and finish quality.
2. **Log_GrLivArea**: Log-transformed above-ground living area.
3. **GarageCars**: Capacity of the garage in car units.
4. **Log_1stFlrSF**: Log-transformed square footage of the first floor.
5. **YearBuilt**: Year of construction.
6. **Log_TotRmsAbvGrd**: Log-transformed total rooms above ground.
7. **Log_House_Age**: Log-transformed age of the house.

---

## Features

### Core Components of the Project:
1. **Data Cleaning**:
   - Imputed missing values using **KNN Imputer** and domain-specific rules.
   - Dropped features with over 80% missing values.
2. **Feature Engineering**:
   - Log transformations applied to reduce skewness.
   - Created new features like **House_Age**.
3. **Exploratory Data Analysis**:
   - Heatmaps, scatter plots, HiPlot visualizations for feature relationships.
4. **Model Building**:
   - Trained advanced models: Ridge, Lasso, Random Forest, and Gradient Boosting.
5. **Real-world Insights**:
   - Evaluated macroeconomic indicators (CPI, Inflation, GDP) and their relationships with housing prices.
6. **Streamlit App**:
   - Interactive user interface for predictions and visual exploration.

---

## Setup and Installation

### Prerequisites
Ensure Python 3.12.4 or later is installed.

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction

