#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import streamlit as st
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix, plot_period_transactions
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
import matplotlib.pyplot as plt

# Import Data
@st.cache
def load_data():
    return pd.read_csv("OnlineRetail.csv", encoding="cp1252")

tx_data = load_data()

# Display title and introduction
st.title('Customer Lifetime Value Calculator')
st.write("This app calculates Customer Lifetime Value (CLV) using the BetaGeoFitter and GammaGammaFitter models from the Lifetimes library.")

# Display sample data
st.subheader('Sample Data:')
st.write(tx_data.head())

# Data preprocessing
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'], format="%m/%d/%Y %H:%M").dt.date
tx_data = tx_data[pd.notnull(tx_data['CustomerID'])]
tx_data = tx_data[tx_data['Quantity'] > 0]
tx_data['Total_Sales'] = tx_data['Quantity'] * tx_data['UnitPrice']

# CLV Calculation
lf_tx_data = summary_data_from_transaction_data(tx_data, 'CustomerID', 'InvoiceDate', monetary_value_col='Total_Sales', observation_period_end='2011-12-9')

# Frequency/Recency Analysis Using the BG/NBD Model
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(lf_tx_data['frequency'], lf_tx_data['recency'], lf_tx_data['T'])

# Visualizations
st.subheader('Frequency/Recency Matrix:')
fig = plt.figure(figsize=(12, 8))
plot_frequency_recency_matrix(bgf)
st.pyplot(fig)

st.subheader('Probability of Being Alive Matrix:')
fig = plt.figure(figsize=(12, 8))
plot_probability_alive_matrix(bgf)
st.pyplot(fig)

# Predict future transaction in next 10 days
t = 10
lf_tx_data['pred_num_txn'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, lf_tx_data['frequency'], lf_tx_data['recency'], lf_tx_data['T']), 2)

# Model Fit Assessment
st.subheader('Model Fit Assessment:')
fig = plt.figure(figsize=(12, 8))
plot_period_transactions(bgf)
st.pyplot(fig)

# Shortlist customers who had at least one repeat purchase with the company
shortlisted_customers = lf_tx_data[lf_tx_data['frequency'] > 0]

# Train gamma-gamma model by taking into account the monetary_value
ggf = GammaGammaFitter(penalizer_coef=0)
ggf.fit(shortlisted_customers['frequency'], shortlisted_customers['monetary_value'])

# Average Transaction Value
lf_tx_data['pred_txn_value'] = round(ggf.conditional_expected_average_profit(
    lf_tx_data['frequency'],
    lf_tx_data['monetary_value']
), 2)

# Calculate Customer Lifetime Value
lf_tx_data['CLV'] = round(ggf.customer_lifetime_value(
    bgf,  # the model to use to predict the number of future transactions
    lf_tx_data['frequency'],
    lf_tx_data['recency'],
    lf_tx_data['T'],
    lf_tx_data['monetary_value'],
    time=12,  # months
    discount_rate=0.01  # monthly discount rate ~ 12.7% annually
), 2)

st.subheader('Top 10 Customers by CLV:')
st.write(lf_tx_data.sort_values(by='CLV', ascending=False).head(10).reset_index())
