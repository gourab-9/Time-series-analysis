import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# Disable the deprecated warning for global pyplot instances
warnings.filterwarnings('ignore', category=FutureWarning)

st.title('Welcome to Time Series Analysis & Visualization')

# Reading the dataset using read_csv
df = pd.read_csv("stock_data.csv", parse_dates=True, index_col="Date")

# Now 'Date' will be your datetime index format
df.index = pd.to_datetime(df.index)

# Deleting column
df.drop(columns=['Unnamed: 0', 'Name'], inplace=True)

analysis_option = st.selectbox("Select an analysis option",
                               ["Plot a Line plot", "Resampling", "Seasonality",
                                "Detecting Stationarity", "Moving Average", "Original Data Vs Differenced Data"])

if analysis_option == "Plot a Line plot":
    st.write("## Plotting Line plot for Time Series data")

    # Plotting Line plot for Time Series data
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='High', label='High Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('Share Highest Price Over Time')
    st.pyplot(plt)

elif analysis_option == "Resampling":
    st.write("## Resampling to monthly frequency, using mean as an aggregation function")

    # Resampling to monthly frequency, using mean as an aggregation function
    df_resampled = df.resample('M').mean()

    sns.set(style="whitegrid")

    # Plotting the 'high' column with seaborn, setting x as the resampled 'Date'
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_resampled, x=df_resampled.index, y='High', label='Month Wise Average High Price', color='blue')

    # Adding labels and title
    plt.xlabel('Date (Monthly)')
    plt.ylabel('High')
    plt.title('Monthly Resampling Highest Price Over Time')
    st.pyplot(plt)
    st.write('We have observed an upward trend in the resampled monthly volume data. An upward trend indicates that, over the monthly intervals, the “high” column tends to increase over time.')

elif analysis_option == "Seasonality":
    st.write("## Seasonality Using Auto Correlation")

    # Plot the ACF
    plt.figure(figsize=(12, 6))
    plot_acf(df['Volume'], lags=40)  # You can adjust the number of lags as needed
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function (ACF) Plot')
    st.pyplot(plt)
    st.write('The presence of seasonality is typically indicated by peaks or spikes at regular intervals, as there are none there is no seasonality in our data.')

elif analysis_option == "Detecting Stationarity":
    st.write("## Detecting Stationarity")

    # Assuming df is your DataFrame
    result = adfuller(df['High'])
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write('Critical Values:', result[4])

elif analysis_option == "Moving Average":
    st.write("## Smoothening the data using Differencing and Moving Average")

    # Differencing
    df['high_diff'] = df['High'].diff()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df['High'], label='Original High', color='blue')
    plt.plot(df['high_diff'], label='Differenced High', linestyle='--', color='green')
    plt.legend()
    plt.title('Original vs Differenced High')
    st.pyplot(plt)

    # Moving Average
    st.write('## Moving Average')
    window_size = 120
    df['high_smoothed'] = df['High'].rolling(window=window_size).mean()

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.plot(df['High'], label='Original High', color='blue')
    plt.plot(df['high_smoothed'], label=f'Moving Average (Window={window_size})', linestyle='--', color='orange')

    plt.xlabel('Date')
    plt.ylabel('High')
    plt.title('Original vs Moving Average')
    plt.legend()
    st.pyplot(plt)

elif analysis_option == "Original Data Vs Differenced Data":
    st.write("## Original Data Vs Differenced Data")

    df['high_diff'] = df['High'].diff()
    # Display the combined DataFrame
    st.write(df[['High', 'high_diff']].head())

    # Assuming df is your DataFrame
    result = adfuller(df['high_diff'].dropna())
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write('Critical Values:', result[4])
    st.write('Based on the ADF Statistici.e < all Critical Values, So, we reject the null hypothesis and conclude that we have enough evidence to reject the null hypothesis. The data appear to be stationary according to the Augmented Dickey-Fuller test.')
    st.write('This suggests that differencing or other transformations may be needed to achieve stationarity before applying certain time series models.')
