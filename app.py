import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

st.title("Solar Power Forecasting")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Solar Power Dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df.dropna()

    st.subheader("Dataset Info")
    st.text(df.info())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Actual Power Output Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["actual_power_output_kW"], ax=ax)
    st.pyplot(fig)

    # Label Encoding
    label_encoder = LabelEncoder()
    df['timestamp'] = label_encoder.fit_transform(df['timestamp'])
    df.drop(columns=["timestamp"], inplace=True)

    st.subheader("Processed Data Preview")
    st.dataframe(df.head())

    # Features and target
    x = df.drop(columns=["actual_power_output_kW"])
    y = df["actual_power_output_kW"]

    # Train-test split
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    mae = mean_absolute_error(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    r2 = r2_score(ytest, ypred)

    st.subheader("Linear Regression Performance")
    st.write("Mean Absolute Error:", mae)
    st.write("Mean Squared Error:", mse)
    st.write("R-squared Score:", r2)

    # XGBoost Model
    model = XGBRegressor()
    model.fit(xtrain, ytrain)
    xgb_score = model.score(xtest, ytest)

    st.subheader("XGBoost Model Performance")
    st.write("XGBoost R-squared Score:", xgb_score)
