import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
import sympy as sp
import sympy.stats as stats
import plotly.graph_objs as go
import pandas_datareader as pdr
import statsmodels.api as sm
from plotnine import *
from mizani.formatters import date_format
from mizani.breaks import date_breaks
import pandas as pd
import matplotlib.pyplot as plt

st.title('_Mean_ Factor (Rewarded Risk)...')

st.write("""

The climate finance literature employ characteristic-sorted portfolio to investigate the relationship between climate risks and cross-sectional stock returns.
Reasoning in terms of portfolio,
         the fundamental debate in the literature is to understand whether climate change risk dynamics 
            are rewarded or unrewarded in the cross-section of stock returns.
         """)


st.write(r"""
Some studies found the evidence of a carbon premium in the cross-section.
         Bolton and Kacperczyk (2021) found that portfolios sorted on the total level and the 
         year-by-year change in emissions were valued at discount. These facts 
         are true regardless of the type of scope emission analysed. Bolton and Kacperczyk (2021)
         further showed that the carbon premium is not linked ot the emission intensity measure.
         """)


# Data for the table
data = {
    "Study": [
        "Bolton and Kacperczyk (2021a)", 
        "Bolton and Kacperczyk (2020)", "Hsu et al. (2020)", 
        "Ardia et al. (2021)", "Cheema-Fox et al. (2021)", "Görgen et al. (2020)",
        "In et al. (2017)", "Pastor et al. (2021)"
    ],
    "Climate risk measure": [
        "Three emission measures", 
        "Three emission measures", "Emission intensity", 
         "Emission intensity", "Two emission measures", 
        "BG scores", 
        "Emission intensity", "E-scores"
    ],
    "Economic rationale": [
        "Transition risk proxies", 
        "Transition risk proxies","Climate policy risk", "Pastor et al. (2020)", "Investor irrationality", 
        "Transition risk proxies", 
        "Investor irrationality", "Pastor et al. (2020)"
    ],

    "Is climate risk priced?": [
        "Yes", "Yes",  
        "Yes", "No", "No", "No", 
        "No",  "No"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the table using Streamlit
st.subheader("Table 1: Climate risk factors and the cross-section of stock returns")
st.write("This table summarizes studies on transition risks factor and their conclusions on the pricing of climate risks.")
st.dataframe(df)

st.write(r"""
         The findings of Bolton and Kacperczyk (2021) are echoed by 
         Bolton and Kacperzcyk (2020) at the international level, although not all countries 
         are pricing cabron risk to the same extent. 
         On the other hand, Hsu $\textit{et al.} (2020)$ discovered that 
         the carbon premium is related to the emission intensity measure.
            """)

st.write(r"""
                  Nevertheless, these findings were not corroborated by the remaining studies in the Table.
""")

st.subheader("Time-Series Regression")

case = st.sidebar.selectbox(
    "Select case:",
    ("g rewarded", "g unrewarded"),
        index=0  # Default index to "Only first factor has a risk premium"
)

st.write(r"""
We go back to our initial model, with one factor describin the excess return. We want to test 
         if there is a risk premia associated with $g$, to determine if:
         """)

st.latex(r"""
        \begin{equation}
r_i = \beta_i (f + \lambda_1) + \gamma_i (g + \lambda_2) + \epsilon_i
\end{equation}
""")

st.write(r"""
         OR:
         """)

st.latex(r"""
        \begin{equation}
r_i = \beta_i (f + \lambda) + \gamma_i g + \epsilon_i
\end{equation}
""")

st.write(r"""
         To do so, we need to run a time-series regression.
         """)



st.write(r"""
         We have $N$ test assets and 2 factors ($f$ and $g$).
""")

T = 10
K = 2


st.write(r"""
         We have the following returns vector for asset $i$ through time:
         """)
# Numerical example
# Initialize variables for the coefficients and the risk premia
alpha = 0.01  # Intercept (constant term)
lambda_f = 0.05  # Risk premium for the first factor (f)
lambda_g = 0.02  # Risk premium for the second factor (g)
noise = np.random.normal(0, 0.005, T)  # Small noise for residuals (epsilon)

# T \times 1, should look like returns for one asset over time
# Generate returns R based on the selected case
if case == "g rewarded":
    # Returns include risk premia for both factors
    R_num = np.array([
        alpha + lambda_f * (0.01 * t) + lambda_g * (0.02 * t) + noise[t - 1] for t in range(1, T + 1)
    ])


elif case == "g unrewarded":
    # Returns only include risk premium for the first factor
    R_num = np.array([
        alpha + lambda_f * (0.01 * t) + noise[t - 1] for t in range(1, T + 1)
    ])


# Show the returns (R) for the selected case
st.write("The observed returns vector for asset $i$ is:")
R_sympy = sp.Matrix(R_num)
st.latex(f"R_i = {sp.latex(R_sympy)}")

st.write(r"""
            We have the following factor realizations (and intercept) through time:
            """)
# T \times K + 1 (for constant alpha) matrix

# Factor realizations depend on the lambdas
np.random.seed(42)
factor_realizations = []
for t in range(1, T + 1):
    if case == "g rewarded":
        f_t = lambda_f + np.random.normal(0, 0.01)  # Around lambda_f with noise
        g_t = lambda_g + np.random.normal(0, 0.01)  # Around lambda_g with noise
        factor_realizations.append([1, f_t, g_t])
    elif case == "g unrewarded":
        f_t = lambda_f + np.random.normal(0, 0.01)  # Around lambda_f with noise
        g_t = np.random.normal(0, 0.01)  # Zero mean but non-zero realizations
        factor_realizations.append([1, f_t, g_t])

# Convert factor realizations to a sympy Matrix
F = sp.Matrix(factor_realizations)

st.latex(f"""F = {sp.latex(F)}""")

st.write(r"""
We can estimate the factor loadings $\beta_i$ and $\gamma_i$ by running the following regression:
         """)

st.latex(r"""
         \begin{equation}
            R = F \begin{bmatrix}
            \alpha_i \\
            \beta_i \\
            \gamma_i \\
            \end{bmatrix}
            + \epsilon
            \end{equation}
            """)

st.write(r"""
         with the following formula:""")

st.latex(r"""
         \begin{equation}
         \begin{bmatrix}
         \alpha_i \\
         \beta_i \\
         \gamma_i \\
         \end{bmatrix}
         = (F^\top F)^{-1} F^\top R
            \end{equation}
            """)


beta_hat_num = (F.T * F).inv() * F.T * R_sympy


st.write(r"""
            The estimated factor loadings are:
            """)
st.latex(f"""\\hat{{B}}_i = {sp.latex(beta_hat_num)}""")


st.subheader("Cross-Sectional Interpretation")

st.write(r"""
We then interpret the regression as a description of the cross section:
         """)

st.latex(r"""
         \begin{equation}
         E(R^e_{i}) = \alpha_i + \beta_i^\top E(f_t)
            \end{equation}
            """)

st.write(r"""
         with $E(f_t)$ the average value of the factor:
         """)

st.latex(r"""
            \begin{equation}
            E(f_t) = \frac{1}{T} \sum_{t=1}^T f_t = \lambda
            \end{equation}
            """)
         
lambda_ = F.T * sp.ones(T, 1) / T

st.latex(f"""\\lambda = {sp.latex(lambda_)}""")

st.write(r"""
         Therefore expected returns are:
         """)

expected_returns = (beta_hat_num.T * lambda_)[0]

st.latex(f"""E(R^e_i) = {sp.latex(expected_returns)}""")


st.write(r"""
         If the factor is a mean factor (rewarded risk), then $\lambda = E(f_t)$,
         meaning the factor's average value (mean) represents the compensation for bearing 
         that risk
         """)

st.write(r"""
         We estimate the slope of the cross-sectional relationship by finding 
         the mean of the factor. This is the essence of a rewarded risk: assets 
         that load on factors with positive means (ie. $\lambda > 0$) are expected to have higher returns.
         """)
         # Number of assets
N = 10

# Generate gamma_i (loadings for the second factor g) values between -1 and 1
gamma_i = np.random.uniform(-1, 1, N)

lambda_val = lambda_[2]  # The lambda value for the second factor g

# Small alpha values for assets
alpha_i = np.random.normal(0, 0.01, N)

# Compute expected returns E(R^e_i) = alpha_i + gamma_i * lambda_g
expected_returns_g = alpha_i + gamma_i * lambda_val

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(gamma_i, expected_returns_g, label="Assets", color="blue")

# Plot the regression line with slope lambda (for γ)
x_vals = np.array([-1, 1])
y_vals = lambda_val * x_vals
ax.plot(x_vals, y_vals, label="Slope λ (for γ)", color="orange", linestyle="--")

# Add a line representing E(g) (assumed to be constant for visualization purposes)
E_g = lambda_val  # The average value of the factor is equal to lambda in this example
ax.axhline(y=E_g, color="gray", linestyle=":", label="E(g)")

# Highlight the point where gamma = 1 and E(R^e) = E(g)
ax.plot(1, E_g, marker='o', markersize=8, color="red", label=r"$(\gamma=1, E(g))$")
ax.annotate(r"$(1, E(g))$", (1, E_g), xytext=(1.1, E_g + 0.01),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Highlight the intercept for a particular asset (choose one asset to demonstrate alpha)
ax.annotate(r"$\alpha_i$", (gamma_i[1], expected_returns_g[1]), xytext=(gamma_i[1] + 0.2, expected_returns_g[1] - 0.02),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Labels and legend
ax.set_xlabel(r"$\gamma_i$")
ax.set_ylabel(r"$E(R^e_i)$")
ax.set_title("Relationship between Factor Loadings for $g$ (γ) and Expected Returns")
ax.legend()
ax.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)




# bmg = pd.read_excel('data/carbon_risk_factor_updated.xlsx', sheet_name='daily', index_col=0)

# if st.button("Cumulative Returns"):
#     bmg_cum = (1 + bmg).cumprod() - 1
#     plot = (ggplot(bmg_cum.reset_index(), aes(x='date', y='BMG')) +
#         geom_line() + 
#         scale_x_date(breaks=date_breaks('1 years'), labels=date_format('%Y')) + 
#         labs(title="Cumulative Returns of BMG Portfolio", x="Date", y="Cumulative Return") +
#         theme(axis_text_x=element_text(rotation=45, hjust=1))
#         )
    

#     st.pyplot(ggplot.draw(plot))

# @st.cache_data
# def load_ff3_data():
#     """Load the Fama-French factors data."""
#     start_date = '2000-01-01'
#     end_date = '2019-06-30'
#     factors_ff3_daily_raw = pdr.DataReader(
#         name="F-F_Research_Data_Factors_daily",
#         data_source="famafrench",
#         start=start_date,
#         end=end_date)[0]
#     factors_ff3_daily = (factors_ff3_daily_raw
#                          .divide(100)
#                          .reset_index(names="date")
#                          .rename(str.lower, axis="columns")
#                          .rename(columns={"mkt-rf": "mkt_excess"}))
#     return factors_ff3_daily

# @st.cache_data
# def compute_rolling_r2(hml, industry_portfolios, rolling_window=126):
#     """Compute rolling R-squared values from regression of HML on industry portfolios."""
#     r2_values = []
#     for i in range(len(hml) - rolling_window + 1):
#         y = hml.iloc[i:i + rolling_window]
#         X = industry_portfolios.iloc[i:i + rolling_window]
#         X = sm.add_constant(X)
#         model = sm.OLS(y, X).fit()
#         r2_values.append(model.rsquared)
#     return np.array(r2_values)

# @st.cache_data
# def compute_rolling_beta(hml, money_industry, rolling_window=126):
#     """Compute rolling beta of HML on Money Industry portfolio."""
#     beta_values = []
#     for i in range(len(hml) - rolling_window + 1):
#         y = hml.iloc[i:i + rolling_window]
#         X = sm.add_constant(money_industry.iloc[i:i + rolling_window])
#         model = sm.OLS(y, X).fit()
#         beta_values.append(model.params[1])  # The slope (beta) coefficient
#     return np.array(beta_values)

# @st.cache_data
# def compute_rolling_volatility(returns, rolling_window=126):
#     """Compute rolling annualized volatility of a portfolio."""
#     # Compute rolling standard deviation of daily returns
#     rolling_volatility = returns.rolling(window=rolling_window).std()
    
#     # Annualize the volatility: Multiply by the square root of 252 (trading days per year)
#     annualized_volatility = rolling_volatility * np.sqrt(252)
    
#     return annualized_volatility


# # Main logic
# factors_ff3_daily = load_ff3_data()
# bmg = pd.read_excel('data/carbon_risk_factor_updated.xlsx', sheet_name='daily', index_col=0)

# # Merge HML and bmg
# data = pd.merge(factors_ff3_daily[['date', 'hml']], bmg, on='date')


# # Run rolling regression and compute R-squared values
# if st.button('Run Rolling Regression and Plot'):
#     r2_values = compute_rolling_r2(data['hml'], data.drop(columns=['date', 'hml']))
    
#     # Add the R-squared values to the data
#     data_rolling_r2 = data.iloc[126 - 1:].copy()
#     data_rolling_r2['r2'] = r2_values

#     # Create the plot
#     plot = (ggplot(data_rolling_r2, aes(x='date', y='r2')) +
#             geom_line(color='blue') +
#             labs(title="126-Day Rolling $R^2$: HML on BMG Portfolio",
#                  x="Date", y="R-squared") +
#             scale_x_datetime(breaks=date_breaks('1 years'), labels=date_format('%Y')) +
#             theme(axis_text_x=element_text(rotation=45, hjust=1)))

#     st.pyplot(ggplot.draw(plot))