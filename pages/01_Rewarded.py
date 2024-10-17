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


st.title('Rewarded Risks')


st.subheader('Characteristics and the Cross Section of Stock Returns')

st.write(r"""
The asset pricing literature has found that some characteristics does a good job in describing the cross-section of stocks returns. 
These characteristics may act as proxies for exposure to an underlying risk factors. It means that 
it helps to identify stocks that are more likely to perform poorly during bad times. 
Because they are seen as riskier, investors require higher expected returns to hold these stocks.
         
To get a sense of the idea, let's see a simple dividend discount model (as in Fama and French 2015).
We can use this model to link firm characteristics—value, profitability, and investment—with stock returns. 
The following equation represents a simplified one-period dividend discount model (DDM):
""")

st.latex(r'''
            \begin{equation}
         M_t = \frac{\mathbb{E}(Y_{t+1}) - \mathbb{E}(dB_{t+1})}{1+r}
        \end{equation}
            ''')

st.write(r"""
Where:
- $M_t$: Market value of equity (value),
- $Y_{t+1}$: Expected future earnings (profitability),
- $dB_{t+1}$: Expected change in book value of equity (investment),
- $r$: Expected return (approximately).

By solving for $r$, we obtain the expected return:
""")


# Define the equation: r = (Y_{t+1} - dB_{t+1}) / M_t - 1
M_t, Y_t1, dB_t1, r = sp.symbols('M_t Y_t1 dB_t1 r')
ddm_eq = sp.Eq(r, (Y_t1 - dB_t1) / M_t - 1)

st.latex(r'''
            \begin{equation}
            r = \frac{Y_{t+1} - dB_{t+1}}{M_t} - 1
            \end{equation}
         ''')


default_M = 20
default_Y_t1 = 30
default_dB_t1 = 5

# User inputs for the equation
M_t_val = st.sidebar.number_input('Case 1: Market value of equity ($M_t$)',min_value=15, max_value=25, value=default_M, step=1)
Y_t1_val = st.sidebar.number_input('Case 2: Expected future earnings ($Y_{t+1}$)',min_value = 25, max_value = 35, value=default_Y_t1, step = 1)
dB_t1_val = st.sidebar.number_input('Case 3: Expected change in book value of equity ($dB_{t+1}$)', value=default_dB_t1)

# Substitute user input values into the equation
ddm_numeric_case_1 = ddm_eq.subs({M_t: M_t_val, Y_t1: default_Y_t1, dB_t1: default_dB_t1})

# Solve for r (expected return)
expected_return_case_1 = sp.solve(ddm_numeric_case_1, r)[0]


st.write(r"""
Playing around with the equation, we can see how different different characteristics may affect the expected returns:
1. **Valuation**: If we hold everything else constant and only vary the market value ($M_t$), 
         a lower $M_t$ (or a higher book-to-market ratio) implies a higher expected return.
         Play with the market value slider to see how it affects the expected return.
""")


# Construct the LaTeX expression using the actual values
latex_eq = r"\begin{equation} r = \frac{" + str(default_Y_t1) + " - " + str(default_dB_t1) + "}{" + str(M_t_val) + "} - 1 =" + str(round(expected_return_case_1,2)) +"\end{equation}" 

# Display the equation in LaTeX format in Streamlit
st.latex(latex_eq)

st.write(r"""
         It doesn't mean that value firms are better than growth firms. It means 
         that value firms are riskier and thus offer higher expected returns.
For example, Zhang (2005) explains the value premium by arguing that value 
         firms are more affected by bad economic times, 
         as their assets are harder to adjust compared to growth firms.
          As a result, value firms tend to offer higher expected returns 
         due to their greater exposure to risks in bad times.
""")

st.write(r"""
2. **Profitability**: If we fix the market value ($M_t$) and 
         only vary the expected future earnings ($Y_{t+1}$), 
         higher expected earnings imply a higher expected return.
         Again, play around with the slider to see how it affects the expected return.
""")


# Substitute user input values into the equation
ddm_numeric_case_2 = ddm_eq.subs({M_t: default_M, Y_t1: Y_t1_val, dB_t1: default_dB_t1})

# Solve for r (expected return)
expected_return_case_2 = sp.solve(ddm_numeric_case_2, r)[0]


# Construct the LaTeX expression using the actual values
latex_eq = r"\begin{equation} r = \frac{" + str(Y_t1_val) + " - " + str(default_dB_t1) + "}{" + str(default_M) + "} - 1 =" + str(round(expected_return_case_2,2)) +"\end{equation}"

# Display the equation in LaTeX format in Streamlit
st.latex(latex_eq)

st.write(r"""
While profitability might intuitively suggest lower risk, 
         firms with higher profitability often have more of their 
         cash flow far into the future, making these cash flows more uncertain. 
         Additionally, profitable firms may attract competition, which can threaten future profit margins and increase risk.
""")

st.write(r"""
3. **Investment**: If we hold the market value and expected earnings constant, 
         higher expected growth in book equity (investment) implies a lower expected return.
         Play with the investment slider to see how it affects the expected return.
""")

# Substitute user input values into the equation
ddm_numeric_case_3 = ddm_eq.subs({M_t: default_M, Y_t1: default_Y_t1, dB_t1: dB_t1_val})

# Solve for r (expected return)
expected_return_case_3 = sp.solve(ddm_numeric_case_3, r)[0]

# Construct the LaTeX expression using the actual values
latex_eq = r"\begin{equation} r = \frac{" + str(default_Y_t1) + " - " + str(dB_t1_val) + "}{" + str(default_M) + "} - 1 =" + str(round(expected_return_case_3,2)) +"\end{equation}"

# Display the equation in LaTeX format in Streamlit
st.latex(latex_eq)

st.write(r"""
This relationship suggests that firms investing 
         heavily to sustain profits may have lower 
         free cash flow available for investors, 
         leading to lower expected returns compared to firms with lower investment needs.
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
         """)

st.write(r"""
         Bolton and Kacperczyk (2020) decomposed the three measures of carbon risk for each type of emissions.
         Notably, as Busch $\textit{et al.}$ (2018) observed, there is little variation 
         in the reported scope 1 and 2 emissions among data providers. 
         """)

st.write(r"""
One must understand how the emission variables are related to the cross-section fo stocks retuns. Bolton and 
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
         They argued that merging the ESG variables between these datasets should minimise the 
         potential self-reporting bias.
         """)

st.write(r"""
         Pastor $\textit{et al.}$ (2021) proxies this overall climate risk exposure by means 
         of the E-scores provided by the MSCI and Sustainalytics databases, arguing that they should capture the 
         different dynamics. Engle $\textit{et al.}$ (2020) 
         constructed E-score measures at the firm level by taking the difference between positive and negative 
         E-scores subcategories. 
         """)

st.subheader('Characteristic-Sorted Portfolio')

st.write(r"""
A common practice in the academic finance literature 
has been to create characteristic portfolios by sorting on 
characteristics positively associated with expected returns.
The resultant portfolios, which go long a portfolio of high characteristic
firms and short a portfolio of low characteristic firms serve as a proxy for 
the risk factor returns (see Fama and French 1992 and 2015).
""")

st.write(r"""
Following Daniel $\textit {et al.}$ (2020), 
we consider a one-period economy with $N$ assets. 
Realized excess returns are given by:
""")

st.latex(r"""
        \begin{equation}
r_i = \beta_i (f + \lambda) + \epsilon_i
\end{equation}
         """)

st.write(r"""
with $r_i$ the excess return of asset $i$, 
$\beta_i$ the factor loading on the rewarded factor $f$,
$\lambda$ the risk premium on $f$ and $\epsilon_i$ the idiosyncratic return. 
We have 
$\mathbb{E}[\epsilon_i] = \mathbb{E}[f] = 0$.

We have $r$ the $N \times 1$ vector of excess returns.
Taking expectations of our first equation, we have:
         """)

st.latex(r"""
         \begin{equation}
    \begin{aligned}
    \mu = \mathbb{E}[r] = \mathbb{E}[\beta (f + \lambda) + \epsilon] \\
    = \beta \mathbb{E}[f + \lambda] + \mathbb{E}[\epsilon] \\
    = \beta \lambda
    \end{aligned}
\end{equation}
         """)

st.write(r"""
         with $\beta$ the $N \times 1$ vector of factor loadings on $f$.
So, expected returns are driven by the exposure to the rewarded factor $f$ 
and the risk premium $\lambda$ on $f$.
Following Daniel $\textit {et al.}$ (2020), we now assume that expected returns 
are described by a single characteristic $x$:
                """)

st.latex(r"""
         \begin{equation}
    \mu = x \lambda_c
\end{equation}
                """)

st.write(r"""
         
where $\lambda_c$ is the risk premium on the characteristic $x$.

To have the two last equations to hold, we must have:
""")

st.latex(r"""
         \begin{equation}
    \begin{aligned}
    \beta \lambda = x \lambda_c \\
    \beta = x \frac{\lambda_c}{\lambda}
    \end{aligned}
\end{equation}
                """)

st.write(r"""
         This equation shows that the characteristic is a perfect 
proxy for the exposure to the rewarded factor. 

We now consider only 6 assets with equal market capitalization.
The assets have characteristics and loadings on the rewarded factor as:
""")

N = 12  # Number of assets

# Predefined fixed values for beta
fixed_beta = [1, 1, 1, 1, 1,1, -1, -1, -1,-1,-1,-1]

# Create a LaTeX table for the beta inputs
beta_latex = r"\begin{array}{|c|c|} \hline Asset & \beta \\ \hline "
for i in range(N):
    beta_latex += f"{i+1} & {fixed_beta[i]} \\\\ \\hline "
beta_latex += r"\end{array}"
st.latex(beta_latex)



# Convert beta inputs to Sympy matrices
beta = sp.Matrix(fixed_beta)

# Portfolio weights based on sorted betas (long the highest, short the lowest)
beta_np = np.array(fixed_beta)

# Get the indices of the sorted beta values
sorted_indices = np.argsort(beta_np)

# Use SymPy's Rational to keep weights as fractions
w_long = sp.Matrix([0]*N)
w_short = sp.Matrix([0]*N)

# Assign long positions (1/3) to the top 3 assets
for idx in sorted_indices[-int(N/2):]:
    w_long[idx] =sp.Rational(1, int(N/2))

# Assign short positions (-1/3) to the bottom 3 assets
for idx in sorted_indices[:int(N/2)]:
    w_short[idx] = sp.Rational(-1, int(N/2))

# Combine long and short positions to form the final weight vector
w = w_long + w_short

# Display the resulting portfolio in LaTeX format
st.write(r"""
        We can construct a portfolio $c$ on the basis of characteristic 
$x$ (or $\beta$, as we have supposed that $x$ is a perfect proxy for $\beta$).
The weights are given by:
         """)
# Prepare weights in LaTeX format as a row vector
weights_latex = r"\begin{bmatrix} "
for i in range(N):
    weights_latex += f"{sp.latex(w[i])} & "
weights_latex = weights_latex[:-2] + r" \end{bmatrix}"  # Remove the last "&" and close the matrix


st.latex(r"""
w_c^T = """ + weights_latex)



# Define priced factor as normal random variable with variance properties
f = stats.Normal('f', 0, sp.symbols('sigma_f'))  # Priced factor f with E[f] = 0 and var(f) = sigma_f^2

# Define idiosyncratic errors epsilon_i as random variables with zero mean and variance sigma_epsilon^2
epsilon = sp.Matrix([stats.Normal(f'epsilon{i+1}', 0, sp.symbols('sigma_epsilon')) for i in range(N)])
# Define symbols for variances of the factors and idiosyncratic error
sigma_f, sigma_epsilon = sp.symbols('sigma_f sigma_epsilon')

# Characteristic premium
lambda_ = sp.symbols('lambda')

# Step 1: Define the portfolio return formula symbolically
portfolio_return = w.dot(beta * (f + lambda_) + epsilon)

st.write(r"""
         We can now compute the return of the portfolio $c$:
                """)


st.latex(r"""
\begin{equation}
         \begin{aligned}
         r_c = w^\top r \\
         = w^\top (\beta (f + \lambda) + \epsilon)
         \end{aligned}
\end{equation}
         """)
st.latex(f"""r_c= {sp.latex(portfolio_return)}""")

# Step 2: Take the expectation using sympy.stats
expected_portfolio_return = stats.E(portfolio_return)

# Contribution from the priced factor f:
# LaTeX: Var_f = (w^\top \beta)^2 \sigma_f^2
variance_f = (w.dot(beta))**2 * sigma_f**2  # Contribution from priced factor f

# Contribution from the idiosyncratic errors:
# LaTeX: Var_\epsilon = w^\top w \times \sigma_\epsilon^2
variance_epsilon = w.dot(w) * sigma_epsilon**2  # Contribution from idiosyncratic errors

# Total variance of the portfolio:
# total_portfolio_variance = variance_f + variance_epsilon
# Define the covariance matrix for the systematic factor
# This is beta * beta.T * sigma_f**2 (N x N matrix)
covariance_matrix_f = beta * beta.T * sigma_f**2

# Define the covariance matrix for idiosyncratic errors (N x N identity matrix scaled by sigma_epsilon**2)
covariance_matrix_epsilon = sigma_epsilon**2 * sp.eye(N)

# Combine the covariance matrices
covariance_matrix = covariance_matrix_f + covariance_matrix_epsilon

# Calculate the total portfolio variance as w.T * covariance_matrix * w
total_portfolio_variance = (w.T * covariance_matrix * w)[0]

# Calculate the Sharpe ratio
sharpe_ratio = expected_portfolio_return / sp.sqrt(total_portfolio_variance)

beta_p = beta.dot(w)

st.write(fr"""
The portfolio returns capture the expected returns 
because it loads on the rewarded factor $f$ with $\beta_c = {sp.latex(w.dot(beta))}$.
The expected return of the portfolio is:
""")


st.latex(r"""
\begin{equation}
            \mathbb{E}[r_c] = w^\top \mu = w^\top \beta \lambda
\end{equation}
         """)

st.latex(f"E[r_c] = {sp.latex(expected_portfolio_return)}")


st.write(r"""
         The variance of the portfolio is:
                """)

st.latex(r"""
\begin{equation}
         \sigma_c^2 = w^\top \Sigma w = w^\top \left( \beta \beta^\top \sigma_f^2 + \sigma_\epsilon^2 I \right) w
\end{equation}
         """)
st.latex(f"\\sigma^2_c = {sp.latex(total_portfolio_variance)}")

st.write(r"""
         which give us the Sharpe ratio of the portfolio:
                """)

st.latex(f"\\text{{Sharpe Ratio}} = {sp.latex(sharpe_ratio)}")

st.subheader('Climate Risk Factor?')


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



# @st.cache_data
# def load_ff5_data():
#     """Load the Fama-French factors data."""
#     start_date = '2000-01-01'
#     end_date = '2019-06-30'
#     factors_ff5_daily_raw = pdr.DataReader(
#         name="F-F_Research_Data_5_Factors_2x3_daily",
#         data_source="famafrench",
#         start=start_date,
#         end=end_date)[0]
#     factors_ff5_daily_raw = (factors_ff5_daily_raw
#                          .divide(100)
#                          .reset_index(names="date")
#                          .rename(str.lower, axis="columns")
#                          .rename(columns={"mkt-rf": "mkt_excess"})
#                          .drop(columns=['rf', 'mkt_excess'])
#                          )
#     return factors_ff5_daily_raw

# @st.cache_data
# def compute_cumulative_returns(df):
#     """Compute cumulative returns for all return columns in the DataFrame."""
#     cumulative_df = df.copy()

#     # Get all columns except the date column
#     return_columns = [col for col in df.columns if col != 'date']

#     # Compute cumulative returns for each column
#     for col in return_columns:
#         cumulative_df[col + '_cumulative'] = (1 + cumulative_df[col]).cumprod() - 1

#     return cumulative_df, return_columns

# # Main logic
# factors_ff3_daily = load_ff5_data()

# # Compute cumulative returns for all factors
# cumulative_factors_df, return_columns = compute_cumulative_returns(factors_ff3_daily)

# # Melt the DataFrame to make it suitable for ggplot
# cumulative_factors_melted = pd.melt(cumulative_factors_df, id_vars=['date'], 
#                                     value_vars=[col + '_cumulative' for col in return_columns],
#                                     var_name='factor', value_name='cumulative_return')

# # Plot cumulative returns
# if st.button('Plot Cumulative Returns of All FF Factors'):
#     plot_cumulative = (ggplot(cumulative_factors_melted, aes(x='date', y='cumulative_return', color='factor')) +
#                        geom_line() +
#                        labs(title="Cumulative Returns of Fama-French Factors",
#                             x="Date", y="Cumulative Return") +
#                        scale_x_datetime(breaks=date_breaks('2 years'), labels=date_format('%Y')) +
#                        theme(axis_text_x=element_text(rotation=45, hjust=1)) +
#                        scale_color_discrete(name="Factors"))

#     st.pyplot(ggplot.draw(plot_cumulative))

st.subheader('Conclusion')

st.write(r"""
So far, climate risks is not a rewarded factor. Most of the literature stop there. 
         """)