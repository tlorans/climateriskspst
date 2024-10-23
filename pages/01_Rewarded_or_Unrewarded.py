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
from regtabletotext import prettify_result


st.title('Climate Risks: Rewarded or Unrewarded?')

st.write(r"""
The asset pricing literature has found that some characteristics does a good job in describing the cross-section of stocks returns. 
These characteristics may act as proxies for exposure to an underlying risk factors. It means that 
it helps to identify stocks that are more likely to perform poorly during bad times. 
Because they are seen as riskier, investors require higher expected returns to hold these stocks.
                  
A common practice in the academic finance literature 
has been to create characteristic portfolios by sorting on 
characteristics positively associated with expected returns.
The resultant portfolios, which go long a portfolio of high characteristic
firms and short a portfolio of low characteristic firms helps
to estimate the associated risk premia (see Fama and French 1992 and 2015).
The characteristic-sorted portfolio is what we call risk factors. A risk factor with a positive risk premium is called a rewarded risk.                  
""")

st.write("""
The climate finance literature employ characteristic-sorted portfolio to investigate the relationship between climate risks and cross-sectional stock returns.
The investigation is two fold:
         1. Identifying what characteristics are the most relevant to capture climate risks.
        2. Understand whether climate change risks are a new rewarded risk to be added to the list of traditional risk factors.
         """)


st.subheader('Climate-Related Characteristics')

st.write(r"""
Similar to traditional firm characteristics, 
         we can expect that firms more exposed to climate risks 
         may also exhibit higher expected returns, 
         as investors require compensation for these additional risks.
         """)


st.write(r"""
Most studies tend to focus on only one of the three firm-level emissions variables,
         and researchers often argue that these variables should proxy the three transition risks drivers.
Hsu $\textit{et al.}$ (2020) constructed a measure of emission intensity 
at the firm level by aggregating plant-level data from the Toxic Release Inventory (TRI) database in the US.
The Trucost and Thomson Reuters' Asset4 ESG databases provide emission data 
         at the aggregated firm level, both for the United States and the entire world.
Moreover, these databases also provide data related to the three different types of scope emissions. 
         Bolton and Kacperczyk (2020) decomposed the three measures of carbon risk for each type of emissions.
         Notably, as Busch $\textit{et al.}$ (2018) observed, there is little variation 
         in the reported scope 1 and 2 emissions among data providers. 
         """)

st.write(r"""
Bolton and 
         Kacperczyk (2020) argued the total amount of emissions should proxy 
         for the long-term company's exposure to transition risks, as it is likely that regulations 
         aimed to curb emissions are targeted more toward these types of fims. The 
         opposite is true for the year-by-year changes in emissions, as this measure should 
         capture the short-term effects of transition risks on stock returns.
         """)

st.write(r"""
The economic rationale behind the emission intensity measure is explained using two different channels.
         Hsu $\textit{et al.}$ (2020) assumed this measure should proxy for the climate policy risk 
         exposure of pollutant firms, so it is allowed to play a similar role as the total amount of firm 
         emissions as in Bolton and Kacperczyk (2021). 
         """)

st.write(r"""
         Pastor $\textit{et al.}$ (2021) proxies this overall climate risk exposure by means 
         of the E-scores provided by the MSCI and Sustainalytics databases, arguing that they should capture the 
         different dynamics. Engle $\textit{et al.}$ (2020) 
         constructed E-score measures at the firm level by taking the difference between positive and negative 
         E-scores subcategories. 
         """)

st.write(r"""
Görgen $\textit{et al.}$ construct a score able to proxy for the 
several transition risk drivers. In particular, they developed 
a "Brown-Green-Score" (BG) which is defined as:
         """)

st.latex(r'''
         \begin{equation}
         BGS_{i,t} = 0.7 \text{Value Chain}_{i,t}
         + 0.15 \text{Adaptability}_{i,t}
         + 0.15 \text{Public Perception}_{i,t}
            \end{equation}
            ''')

st.write(r"""
The variables $\text{Value Chain}_{i,t}$,
$\text{Adaptability}_{i,t}$ and $\text{Public Perception}_{i,t}$ are
         proxies for the terms policy risk, technology risk and 
         preference risk, respectively.
         To build the measure, they relied on 10 different ESG variables, retrieved 
         from four different data providers.

         """)

st.subheader('A Climate Risks Factor?')

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
         Nevertheless, these findings were not corroborated by the remaining studies in the Table.
            """)

st.write(r"""
We are going to focus on the BMG factor from Gorgen et al. (2020).
First, let's have a look at the cumulative returns of the BMG portfolio.
It seems to be negative for the sample period considered. This is an 
unexpected result, as we would expect a positive return for a rewarded risk.
Brown firms should be riskier and thus offer higher expected returns. It seems 
not to be the case here. 
Thus, first, we do no have the expected sign for the eventually associated risk premia.        
         """)

bmg = pd.read_excel('data/carbon_risk_factor_updated.xlsx', sheet_name='daily', index_col=0)

bmg_cum = (1 + bmg).cumprod() - 1
plot = (ggplot(bmg_cum.reset_index(), aes(x='date', y='BMG')) +
    geom_line() + 
    scale_x_date(breaks=date_breaks('1 years'), labels=date_format('%Y')) + 
    labs(title="Cumulative Returns of BMG Portfolio", x="", y="Cumulative Return") +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
    )


st.pyplot(ggplot.draw(plot))

st.write(r"""
Pastor et al. (2021, 2022) and Ardia et al. (2021) found an interesting 
explanation to this unexpected sign of the BMG factor. They argue that
estimating the expected returns based on realized returns may lead to bias, especially in the case of climate risks.
In an event of unexpected change in climate concerns, we may observe positive 
unexpected returns for green stocks (outperformance) and negative unexpected returns for brown stocks (underperformance).
Estimating the risk premia as the simple average of the realized returns may lead to a negative risk premia for the BMG factor 
if the sample period is characterized by a strengthening of climate concerns.
         """)

st.write(r"""
A second simple test is to check if the average return of the BMG portfolio is different from zero.
The results indicate that we cannot reject the null hypothesis of average 
returns being equal to zero. 
         """)

model_fit = (sm.OLS.from_formula(
    formula="BMG ~ 1",
    data=bmg
)
  .fit(cov_type="HAC", cov_kwds={"maxlags": 6})
)

st.write(model_fit.summary())


st.subheader('What If Rewarded Risks Load on Climate Risks?')


st.write(r"""
We have seen that the BMG factor may be not a rewarded risk factor, as of today.
As we have seen in the introduction, the mean-variance efficient portfolio should load on rewarded risks, 
but diversify away from unrewarded risks. The problem is if rewarded risks factor themselves load on unrewarded risks.
         In that case, our portfolio will indirectly load on unrewarded risks because rewarded risks load on them!
This is what Daniel $\textit{et al.}$ (2020) have shown: characteristics-sorted portfolio loads on unrewarded 
risks as long as there exist a correlation betwen the characteristics and the loadings on the unrewarded risks.
            """)

st.write(r"""
         The proportion of the variance of the rewarded factor explained by the unrewared factor is given by the $R^2$:
         """)

st.latex(r'''
\begin{equation}
         R^{2} = \frac{\beta^2 \sigma_g^2}{\sigma_f^2}
            \end{equation}
         ''')

gamma_f_default = 0.1
sigma_g_default = 0.1  # Default to 1
sigma_f = .1  # Default to 1
beta_f = st.sidebar.slider("$\\beta$", 0.0, 1.0, gamma_f_default, 0.1)
sigma_g = st.sidebar.slider("$\sigma_g$", 0.1, 1.0, sigma_g_default, 0.1)

R_squared = (beta_f ** 2 * sigma_g ** 2) / (sigma_f ** 2)

st.latex(f"R^2 = {sp.latex(round(R_squared, 2))}")




st.write(r"""
We can now investigate if any of the traditional Fama-French factors loads on the BMG factor.
         """)

@st.cache_data
def load_ff5_data():
    """Load the Fama-French factors data."""
    start_date = '2000-01-01'
    end_date = '2019-06-30'
    factors_ff3_daily_raw = pdr.DataReader(
        name="F-F_Research_Data_5_Factors_2x3_daily",
        data_source="famafrench",
        start=start_date,
        end=end_date)[0]
    factors_ff3_daily = (factors_ff3_daily_raw
                         .divide(100)
                         .reset_index(names="date")
                         .rename(str.lower, axis="columns")
                         .rename(columns={"mkt-rf": "mkt_excess"}))
    return factors_ff3_daily

bmg = pd.read_excel('data/carbon_risk_factor_updated.xlsx', sheet_name='daily', index_col=0)
ff5_data = load_ff5_data()
data = pd.merge(ff5_data, bmg, on='date')

# st.write(ff3_data.head())

@st.cache_data
def compute_rolling_beta(f, g, rolling_window=126):
    """Compute rolling beta of HML on Money Industry portfolio."""
    beta_values = [np.nan] * (rolling_window - 1)  # Prepend NaN for the first observations
    for i in range(len(f) - rolling_window + 1):
        y = f.iloc[i:i + rolling_window]
        X = sm.add_constant(g.iloc[i:i + rolling_window])
        model = sm.OLS(y, X).fit()
        beta_values.append(model.params[1])  # The slope (beta) coefficient
    return np.array(beta_values)

# Assuming 'data' is your DataFrame with columns ['date', 'hml', 'BMG']
beta_values = (data 
               .assign(
                   beta_hml = compute_rolling_beta(data['hml'], data['BMG']),
                    beta_smb = compute_rolling_beta(data['smb'], data['BMG']),
                    beta_rmw = compute_rolling_beta(data['rmw'], data['BMG']),
                    beta_cma = compute_rolling_beta(data['cma'], data['BMG']),
                    beta_mkt = compute_rolling_beta(data['mkt_excess'], data['BMG'])
            )
            .dropna()
)

# Create a long-form DataFrame to plot both gamma_hml and gamma_smb
beta_long = pd.melt(beta_values, id_vars=['date'], value_vars=['beta_hml', 'beta_smb', 'beta_rmw', 'beta_cma', 'beta_mkt'],
                     var_name='Factor', value_name='Gamma')

# Plot both gamma_hml and gamma_smb
plot_values = (
    ggplot(beta_long.query('date < "01-01-2019"'), aes(x='date', y='Gamma', color='Factor')) +   
    geom_line() +
    labs(title="Rolling $\\beta$ of Rewarded Factors on BMG Factor",
         x="", y="$\\beta$") +
    scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)

# Display the plot in Streamlit
st.pyplot(ggplot.draw(plot_values))

st.write(r"""
We are now going to investigate the magnitude of it, such as the $\sigma_g^2$.
         """)

@st.cache_data
def compute_rolling_volatility(returns, rolling_window=126):
    """Compute rolling annualized volatility of a portfolio."""
    # Compute rolling standard deviation of daily returns
    rolling_volatility = returns.rolling(window=rolling_window).std()
    
    # Annualize the volatility: Multiply by the square root of 252 (trading days per year)
    annualized_volatility = rolling_volatility * np.sqrt(252)
    
    return annualized_volatility

# Select the "Money Industry" portfolio (replace 'money' with actual column name)
bmg_returns = data['BMG']  # Replace 'money' with the actual column name

# Compute rolling annualized volatility (252-day window)
annualized_volatility = compute_rolling_volatility(bmg_returns)

# Prepare data for plotting
data_volatility = data.copy()
data_volatility['annualized_volatility'] = annualized_volatility

# Create the plot for annualized volatility
plot_volatility = (ggplot(data_volatility.query('date < "01-01-2019"'), aes(x='date', y='annualized_volatility')) +
                    geom_line(color='green') +
                    labs(title="126-Day Rolling Annualized Volatility: BMG",
                        x="Date", y="Annualized Volatility") +
                    scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
                    theme(axis_text_x=element_text(rotation=45, hjust=1)))

st.pyplot(ggplot.draw(plot_volatility))

st.write(r"""
         Results in the $R^2$:
            """)

@st.cache_data
def compute_rolling_r2(f, g, rolling_window=126):
    """Compute rolling R-squared values from regression of HML on industry portfolios."""
    r2_values = [np.nan] * (rolling_window - 1)  # Prepend NaN for the first observations
    for i in range(len(f) - rolling_window + 1):
        y = f.iloc[i:i + rolling_window]
        X = g.iloc[i:i + rolling_window]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        r2_values.append(model.rsquared)
    return np.array(r2_values)


# Assuming 'data' is your DataFrame with columns ['date', 'hml', 'BMG']
r_values = (data 
               .assign(
                   r_hml = compute_rolling_r2(data['hml'], data['BMG']),
                     r_smb = compute_rolling_r2(data['smb'], data['BMG']),
                     r_rmw = compute_rolling_r2(data['rmw'], data['BMG']),
                        r_cma = compute_rolling_r2(data['cma'], data['BMG']),
                        r_mkt = compute_rolling_r2(data['mkt_excess'], data['BMG'])
            )
            .dropna()
)

# Create a long-form DataFrame to plot both gamma_hml and gamma_smb
r_values_long = pd.melt(r_values, id_vars=['date'], value_vars=['r_hml', 'r_smb', 'r_rmw', 'r_cma', 'r_mkt'],
                     var_name='Factor', value_name='R_squared')

# Plot both gamma_hml and gamma_smb
plot_r_values = (
    ggplot(r_values_long.query('date < "01-01-2019"'), aes(x='date', y='R_squared', color='Factor')) +   
    geom_line() +
    labs(title="Rolling $R^2$ of Rewarded Factors on BMG Factor",
         x="", y="$R^2$") +
    scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)

# Display the plot in Streamlit
st.pyplot(ggplot.draw(plot_r_values))


st.subheader("Portfolio Implications: an Hidden Risk")  

st.write(r"""What does it means for investors portfolio?
         Let's assume that the investors use the Fama-French five-factors to build their portfolios.""")

st.write(r"""We have the risk premia of the five factors:""")

st.latex(r'''
         \lambda = \begin{bmatrix}
            \lambda_{smb} \\
            \lambda_{hml} \\
            \lambda_{rmw} \\
            \lambda_{cma} \\
            \lambda_{mkt}
            \end{bmatrix}
            ''')

st.write(r"""Let's assume the following values for the risk premia:""")

# Define symbols for SymPy
k = 5  # Number of factors

lambda_factors = sp.Matrix([0.02, 0.03, 0.04, 0.01, 0.05])
lambda_factors_latex = sp.latex(lambda_factors)
st.latex(fr'''\lambda = {lambda_factors_latex}''')

st.write(r"""The covariance matrix for the factors, assumed to be uncorrelated:""")

st.latex(r'''
\begin{equation}
         \Omega = \begin{bmatrix}
         \sigma_{smb}^2 & 0 & 0 & 0 & 0 \\
        0 & \sigma_{hml}^2 & 0 & 0 & 0 \\
        0 & 0 & \sigma_{rmw}^2 & 0 & 0 \\
        0 & 0 & 0 & \sigma_{cma}^2 & 0 \\
        0 & 0 & 0 & 0 & \sigma_{mkt}^2
         \end{bmatrix}
            \end{equation}
         ''')

st.write(r"""Let's now express the variance of the rewarded factors as a combination of two components:
1. A part that is uncorrelated with the unrewarded risk,
2. A part that is correlated with the unrewarded risk (expressed through the loading or beta).

This way, we can decompose the variance of the rewarded factor as follows:""")

st.latex(r'''
\begin{equation}
\sigma_f^2 = (1 - \beta^2) \sigma_f^2 + \beta^2 \sigma_g^2
\end{equation}
''')

st.write(r"""In this equation:
- $\sigma_f^2$ is the total variance of the rewarded factor,
- $(1 - \beta^2) \sigma_f^2$ represents the portion of the variance of the rewarded factor that is uncorrelated from the unrewarded risk,
- $\beta^2 \sigma_g^2$ represents the portion of the variance that is correlated with the unrewarded risk.
            """)


# Original variances of the Fama-French factors (you can change these values if needed)
sigma_factors = sp.Matrix([0.1, 0.15, 0.2, 0.12, 0.18])

# Decompose the variance for each factor, incorporating the effect of the unrewarded risk
omega_elements = [(1 - beta_f**2) * sigma_factors[i]**2 + beta_f**2 * sigma_g**2 for i in range(k)]

# Create the Omega matrix (covariance matrix of the factors)
Omega = sp.diag(*omega_elements)

# Display the Omega matrix
st.latex(r'\Omega = ' + sp.latex(Omega))

st.write(r"""We assume the investor) can invest in 5 assets, each one perfectly correlated
         with one of the Fama-French factors. The matrix of factor loadings is the identity matrix:""")
# one asset loads perfectly on one rewarded factor, 0 on the others (Identity matrix)
B = sp.Matrix(np.eye(k))

# Display the B matrix
st.latex(r'B = ' + sp.latex(B))

st.write(r"""Ignoring idiocyncratic risk, the covariance matrix of the asset returns is therefore the covariance matrix of factors:""")

# Compute the covariance matrix of asset returns
Sigma = B * Omega * B.T

# Display the covariance matrix of asset returns
st.latex(r'\Sigma = ' + sp.latex(Sigma))

st.write(r"""The expected return of the portfolio is given by the factor loadings times the risk premia, which is the same as the expected return of the factors:""")

# Compute the expected return of the portfolio
mu = B * lambda_factors

# Display the expected return of the portfolio
st.latex(r'\mu = ' + sp.latex(mu))

st.write(r"""We therefore have the optimal portfolio weights:""")

# Compute the optimal portfolio weights
numerator = Sigma.inv() * mu
denominator = sp.Matrix([1] * k).T * Sigma.inv() * mu

# Compute the optimal portfolio weights
w_star = numerator / denominator[0]

# Display the optimal portfolio weights
st.latex(r'w^* = ' + sp.latex(w_star))


portfolio_expected_return = w_star.T * mu   # Portfolio expected return
portfolio_variance = w_star.T * Sigma * w_star  # Portfolio variance

# Extract scalar values from the 1x1 matrices
portfolio_expected_return_scalar = round(portfolio_expected_return[0] * 100,2)
portfolio_variance_scalar = round(portfolio_variance[0] * 100,2)

# Compute Sharpe ratio
portfolio_sharpe_ratio = portfolio_expected_return_scalar / sp.sqrt(portfolio_variance_scalar)
# Compute portfolio betas and R-squared as before
portfolio_betas = B.T * w_star

# Compute the overall R-squared of the portfolio
numerator_r_squared_c = (w_star.T * B * Omega * B.T * w_star)[0]  # Extract scalar
denominator_r_squared_c = (w_star.T * Sigma * w_star)[0]  # Extract scalar

# Compute R-squared (overall)
r_squared_c = numerator_r_squared_c / denominator_r_squared_c * 100

# Compute R-squared for each factor
r_squared_factors = []
for j in range(B.shape[1]):  # Loop through each factor
    numerator_r_squared_j = (w_star.T * B[:, j] * B[:, j].T * Omega[j, j] * w_star)[0]  # Extract scalar
    denominator_r_squared_j = denominator_r_squared_c  # The denominator is the same
    r_squared_j = numerator_r_squared_j / denominator_r_squared_j
    r_squared_factors.append(r_squared_j)

# Convert values to latex-friendly strings
r_squared_c_latex = sp.latex(round(r_squared_c,2))
r_squared_factors_latex = [sp.latex(round(r2 * 100,2)) for r2 in r_squared_factors]

# Display the results in a LaTeX table

table_latex = r"\begin{array}{|c|c|} \hline \text{Metric} & \text{Value} \\ \hline"
table_latex += rf"\text{{Expected Return}} & {sp.latex(portfolio_expected_return_scalar)} \\"
table_latex += rf"\text{{Volatility}} & {sp.latex(sp.sqrt(portfolio_variance_scalar))} \\"
table_latex += rf"\text{{Sharpe Ratio}} & {sp.latex(portfolio_sharpe_ratio)} \\"
table_latex += rf"\beta_{{smb}} & {sp.latex(round(portfolio_betas[0], 2))} \\"
table_latex += rf"\beta_{{hml}} & {sp.latex(round(portfolio_betas[1], 2))} \\"
table_latex += rf"\beta_{{rmw}} & {sp.latex(round(portfolio_betas[2], 2))} \\"
table_latex += rf"\beta_{{cma}} & {sp.latex(round(portfolio_betas[3], 2))} \\"
table_latex += rf"\beta_{{mkt}} & {sp.latex(round(portfolio_betas[4], 2))} \\"
table_latex += rf"\text{{R}}^2_{{smb}} & {r_squared_factors_latex[0]} \\"
table_latex += rf"\text{{R}}^2_{{hml}} & {r_squared_factors_latex[1]} \\"
table_latex += rf"\text{{R}}^2_{{rmw}} & {r_squared_factors_latex[2]} \\"
table_latex += rf"\text{{R}}^2_{{cma}} & {r_squared_factors_latex[3]} \\"
table_latex += rf"\text{{R}}^2_{{mkt}} & {r_squared_factors_latex[4]} \\"
table_latex += r"\hline"
table_latex += r"\end{array}"

st.write("The tangency portfolio displays the following statistics:")
st.latex(table_latex)

st.write("The higher the loading of the rewarded factor on the unrewarded factor, the higher the variance of the rewarded factor that is correlated with the unrewarded factor. This leads to a higher overall variance of the portfolio, and a lower Sharpe ratio. The portfolio is therefore less efficient, as it is exposed to unrewarded risks.")


