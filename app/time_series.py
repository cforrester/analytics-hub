import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run(df):
    st.header("Time Series Forecasting")
    if df is None:
        st.info("Please upload a time series dataset (with a date/time column).")
        return
    columns = df.columns.tolist()
    date_col = st.selectbox("Select the date/time column", columns)
    value_col = st.selectbox("Select the value column to forecast", [col for col in columns if col != date_col])
    model_type = st.radio("Forecasting Model", ["Prophet", "ARIMA"], index=0)
    period = st.number_input("Forecast horizon (number of periods to predict)", min_value=1, value=10)
    if st.button("Generate Forecast"):
        # Prepare dataframe for forecasting
        df_ts = df.copy()
        try:
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        except Exception as e:
            st.error(f"Failed to parse dates: {e}")
            return
        df_ts = df_ts.sort_values(date_col)
        df_ts = df_ts[[date_col, value_col]].dropna()
        if df_ts.empty:
            st.error("No data available for forecasting after preprocessing.")
            return
        if model_type == "Prophet":
            # Prophet expects columns: ds (date), y (value)
            from prophet import Prophet
            df_prophet = df_ts.rename(columns={date_col: "ds", value_col: "y"})
            model = Prophet()
            try:
                model.fit(df_prophet)
            except Exception as e:
                st.error(f"Prophet model fitting failed: {e}")
                return
            future = model.make_future_dataframe(periods=int(period))
            forecast = model.predict(future)
            st.subheader("Forecast results (last few periods):")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            # Plot forecast
            fig1 = model.plot(forecast)
            st.pyplot(fig1)
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
        else:  # ARIMA
            from statsmodels.tsa.arima.model import ARIMA
            # Set index as date for ARIMA
            ts_data = df_ts.set_index(date_col)[value_col]
            try:
                model = ARIMA(ts_data, order=(1,1,1))
                model_fit = model.fit()
            except Exception as e:
                st.error(f"ARIMA model fitting failed: {e}")
                return
            forecast = model_fit.forecast(steps=int(period))
            st.subheader("Forecast values:")
            st.write(forecast)
            # Plot actual vs forecast
            fig, ax = plt.subplots()
            ts_data.plot(ax=ax, label="Historical")
            forecast.plot(ax=ax, label="Forecast", style='--')
            ax.legend()
            st.pyplot(fig)
