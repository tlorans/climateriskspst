import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt


st.write("Page under construction")
# # FastAPI endpoint URL
# base_url = "http://127.0.0.1:8000/api/v1"

# # Function to get all fund names
# def get_fund_names():
#     try:
#         response = requests.get(f"{base_url}/funds/")
#         response.raise_for_status()
#         fund_names = response.json()
#         return fund_names
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching fund names: {e}")
#         return []

# # Function to get fund holdings
# def get_fund_holdings(fund_id):
#     try:
#         response = requests.get(f"{base_url}/funds/{fund_id}/holdings")
#         response.raise_for_status()
#         holdings = response.json()
#         return holdings
#     except requests.exceptions.RequestException as e:
#         # st.error(f"Error fetching fund holdings: {e}")
#         return []

# # Function to get fund true returns
# def get_fund_true_returns(fund_id):
#     try:
#         response = requests.get(f"{base_url}/funds/{fund_id}/total_returns")
#         response.raise_for_status()
#         true_returns = response.json()
#         return true_returns
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching fund true returns: {e}")
#         return []

# st.title("Portfolio Construction")

# # Get all fund names
# fund_names = get_fund_names()
# selected_fund_name = st.selectbox("Select a fund", fund_names)

# if selected_fund_name:
#     # Fetch fund details
#     response = requests.get(f"{base_url}/funds/{selected_fund_name}")
#     fund = response.json()
#     fund_id = fund['id']

#     # st.subheader("Fund Details")
#     # st.write(f"Name: {fund['name']}")
#     # st.write(f"Fund Share Class ID: {fund['fund_share_class_id']}")

#     # Fetch and display fund holdings
#     holdings = get_fund_holdings(fund_id)
#     st.subheader("Fund Holdings")
#     if holdings:
#         holdings_df = pd.DataFrame(holdings)
#         st.dataframe(holdings_df)

#     # Fetch and plot fund true returns
#     true_returns = get_fund_true_returns(fund_id)
#     st.subheader("Fund Total Returns")
#     if true_returns:
#         returns_df = pd.DataFrame(true_returns)
#         returns_df['date'] = pd.to_datetime(returns_df['date'])
#         returns_df = returns_df.set_index('date')
        
#         st.line_chart(returns_df['total_return'])
