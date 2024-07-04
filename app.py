import streamlit as st

intro_page = st.Page("introduction.py", title = "Climate Risks and Equity Portfolio")
uncertainty_page = st.Page("uncertainty.py", title = "Climate Change Uncertainty")
capm_page = st.Page("capm.py", title = "CAPM with Climate Risks")
portfolio_page = st.Page('portfolio.py', title = "Portfolio Construction")

pg = st.navigation([intro_page,
                    uncertainty_page, 
                    capm_page,
                    portfolio_page])
st.set_page_config(page_title="Climate Risks")
pg.run()

# # Load the initial data
# stocks_data = pd.read_csv('data/model.csv')
# portfolios_data = pd.read_csv('data/fund_compositions.csv')

# merged_data = portfolios_data.merge(stocks_data, on='esa_id', how='left').dropna(subset=['Loss'])

# # Application Title
# st.title('Climate Risks')

# # Sidebar for navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Select a page:", ["From Climate Change Uncertainty to Ambiguity",
#                                         "CAPM With Hereogeneous Beliefs about Climate Risks",
#                                         "Climate Transition Risk Beta",
#                                         "Portfolio Construction"])

# if page == "From Climate Change Uncertainty to Ambiguity":
#     uncertainty.show_page()

# if page == "CAPM With Hereogeneous Beliefs about Climate Risks":


# if page == "Climate Transition Risk Beta":
#     st.title("Climate Transition Risk Beta")

#     st.markdown("""

#         As we have seen in PST (2021), even in the presence of ambiguity about the probability distribution of climate risks,
#         investors can still make portfolio decisions based on their perception of climate risks. To align their portfolios with their beliefs about climate risks,
#         investors can form a climate risks hedging portfolio based on the sensitivities of stocks with an unanticipated climate risks shock. 
#         In this application, we focus on the case of transition risk.
                
#         Realization of an 
#         unanticipated changes in mitigation policies may have "winners" or "losers" 
#         in the stock market. 
#         For example, companies that are heavily dependent on fossil fuels
#         may be negatively exposed to transition risk, while renewable energy companies 
#         may benefit from it.
#         For investors, hedging transition risk would involve to be long stocks that do well in 
#         period of realization of abrupt change in mitigation policies, 
#         and short stocks that do poorly in this period.
#         Approaches in the literature differ in how they identify stocks to long and short
#         in the hedging portfolio. 
#         Alekseev et al. (2022) 
#         identify two main approaches in the literature to identify stocks to long and short in the hedging portfolio:
#         (i) the narrative approach that relies on economic reasoning , and
#         (ii) the mimicking approach, that relies on statistical methods (see Alekseev 
#         et al. (2022) for example).
#         In this paper, we propose a third approach, that relies on a climate stress-test (an example is 
#         Allen et al., 2020
#         methodology at stock level. This approach 
#         uses climate scenarios outcomes
#         from Integrated-Assessment Models (IAMs), to 
#         project stocks' cash flows and valuations under different state of world.
#         This scenario-based approach
#         allows a "disciplined" way to form a priori
#         economic reasoning on stocks' exposure to climate transition risk.
            
#         """)
    
#     # Load data
#     # betas = pd.read_csv('data/betas.csv')
#     loss = pd.read_csv('data/model.csv')
#     infos_equities = pd.read_csv('data/list_ids.csv')[['esa_id', 'name']]

#     # Merge relevant data for the app
#     df = pd.merge(loss, infos_equities, on='esa_id', how='left')

#     climate_models = ["GCAM 6.0 NGFS", 
#                 "MESSAGEix-GLOBIOM 1.1-M-R12",
#                 "REMIND-MAgPIE 3.2-4.6"]
    
#     # Stock selector
#     stock_names = df['name'].unique()
#     stock_names.sort() 
#     selected_stock = st.selectbox('Select a stock to view data:', stock_names)
#     selected_models = st.multiselect('Select a Model', climate_models, default=climate_models[0])


#     filtered_data = df[(df['name'] == selected_stock) & (df['Model'].isin(selected_models))]

#     # Filter out rows with 'Current Policies' scenario
#     loss_data = filtered_data[filtered_data['Scenario'] != 'Current Policies']

#     # Calculate the interaction term
#     # loss_data['Interaction Term'] = loss_data['Loss'] - (loss_data['GrowthFactorLoss'] + loss_data['CarbonPriceLoss'])

#     loss_data.rename(columns={'GrowthFactorLoss': 'Growth Impact',
#                             'CarbonPriceLoss': 'Carbon Price Impact',
#                             'InteractionTerm': 'Interaction Term'}, inplace=True)
#     # Melt the data for plotting
#     loss_data_melted = loss_data.melt(id_vars=['Scenario', 'Model'], 
#                                     value_vars=['Growth Impact', 'Carbon Price Impact',
#                                                 'Interaction Term'],
#                                     var_name='Loss Component', value_name='Value')

#     loss_data_melted.drop_duplicates(inplace=True)
#     fig = px.bar(loss_data_melted, x='Scenario', y='Value', color='Loss Component', 
#                 title='Loss Components per Scenario',
#                 labels={'Value': 'Loss'},
#                 barmode='relative',
#                 hover_data=['Model'])

#     # fig.update_yaxes(visible=False)
#     st.dataframe(loss_data_melted)


#     st.plotly_chart(fig, use_container_width=True)

    
# if page == "Portfolio Construction":
#     st.title("Portfolio Construction")
    

#     # User input for selecting a fund
#     funds = merged_data['fund_name'].unique()
#     fund = st.selectbox('Select fund:', funds)
    
#     # User input for selecting a scenario and model
#     scenario = st.selectbox('Select scenario:', stocks_data['Scenario'].unique())
#     model = st.selectbox('Select model:', stocks_data['Model'].unique())

#     # Input for the percentage of worst issuers to exclude
#     exclusion_rate = st.slider('Percentage of Worst Issuers to Exclude (%)', 0, 100, 20, 5) / 100

#     if fund:
#         fund_data = merged_data[(merged_data['fund_name'] == fund) & 
#                                 (merged_data['Scenario'] == scenario) & 
#                                 (merged_data['Model'] == model)]
        
#         if not fund_data.empty:
#             # Sort by Loss and eliminate the worst performing assets
#             fund_data = fund_data.sort_values(by='Loss', ascending=False)
#             n = fund_data.esa_id.nunique()
#             m = int(n * exclusion_rate)
#             # print(m)
#             if m > 0:
#                 threshold_loss = fund_data['Loss'].iloc[m]
#             else:
#                 threshold_loss = float('inf')  # No exclusion if m == 0

#             # print(exclusion_rate)
#             # print(threshold_loss)
#             # we don't cover every stock, so at the moment normalize the weights to 100
#             fund_data['weight'] = fund_data['weight'] / fund_data['weight'].sum() * 100
#             # Re-weight the remaining assets
#             fund_data['new_weight'] = fund_data.apply(
#                 lambda row: row['weight'] if row['Loss'] <= threshold_loss else 0, axis=1
#             )

#             # Normalize the weights
#             total_weight = fund_data['new_weight'].sum()
#             fund_data['new_weight'] = fund_data['new_weight'] / total_weight * 100

#             # Calculate active share
#             fund_data['Deviation'] = fund_data['new_weight'] - fund_data['weight']
#             # Calculate initial and optimized portfolio exposure to climate risk
#             initial_exposure = (fund_data['weight'] / 100 * fund_data['Loss']).sum()
#             optimized_exposure = (fund_data['new_weight'] / 100 * fund_data['Loss']).sum()
#             # Bar plot of the active share
#             fig = px.bar(fund_data, x='name', y='Deviation', color='Loss',
#                         title='Deviation of New Portfolio',
#                         labels={'Deviation': 'Deviation', 'name': 'Stock', 'Loss': 'Loss'},
#                         hover_data=['Loss', 'weight', 'new_weight'])
#             st.plotly_chart(fig)

#             # Bar plot of initial and optimized portfolio exposure to climate risk
#             fig_exposure = go.Figure(data=[
#                 go.Bar(name='Initial Loss', x=['Initial Portfolio'], y=[initial_exposure], marker_color='blue'),
#                 go.Bar(name='New Loss', x=['New Portfolio'], y=[optimized_exposure], marker_color='green')
#             ])

#             fig_exposure.update_layout(
#                 title='Portfolio Exposure to Climate Transition Risk',
#                 xaxis_title='Portfolio',
#                 yaxis_title='Loss',
#                 template='plotly_white',
#                 barmode='group'
#             )
#             st.plotly_chart(fig_exposure)

#             st.dataframe(fund_data[['name', 'Scenario', 'Model', 'weight', 'new_weight', 'Deviation', 'Loss']])
#         else:
#             st.error('No data found for the selected fund, scenario, and model.')
#     else:
#         st.write("Please select a fund, scenario, and model to display the exposure.")
