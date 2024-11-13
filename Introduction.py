import streamlit as st

st.title('Climate Risks with Python')

st.write(
    """
    Amenc et al. (2014) highlight the two main challenges involved in a sound approach to equity investing according to asset pricing theory: 
    (i) efficiently capturing rewarded factors, and (ii) diversifying away—i.e., reducing exposure to—unrewarded risks. As climate finance 
    is a burgeoning field, the question of how to articulate climate-related risks into this well-recognized investing problem is an open one. 
    To do so, one should first determine in which box climate-related risks may belong: rewarded or unrewarded risks.

    Climate finance literature has investigated whether climate-related risks constitute a new source of rewarded risks. 
    The debate is still ongoing, as no consensus on the topic has been reached yet, as shown in the table below.
    """
)


html_table = """
<table style="width:100%; text-align:left; border-collapse:collapse;">
    <thead>
        <tr style="border-bottom: 2px solid #000;">
            <th>Study</th>
            <th>Climate risk measure</th>
            <th>Economic rationale</th>
            <th>Is climate risk priced?</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Bolton et al. (2020)</td>
            <td>Three emission measures</td>
            <td>Transition risk proxies</td>
            <td>Yes</td>
        </tr>
        <tr>
            <td>Bolton et al. (2021)</td>
            <td>Three emission measures</td>
            <td>Transition risk proxies</td>
            <td>Yes</td>
        </tr>
        <tr>
            <td>Bolton et al. (2023)</td>
            <td>Carbon emissions</td>
            <td>Transition risk proxies</td>
            <td>Yes</td>
        </tr>
        <tr>
            <td>Hsu et al. (2023)</td>
            <td>Emission intensity</td>
            <td>Climate policy risk</td>
            <td>Yes</td>
        </tr>
        <tr>
            <td>Ardia et al. (2023)</td>
            <td>Climate news index</td>
            <td>Pastor & Stambaugh (2021)</td>
            <td>Yes</td>
        </tr>
        <tr>
            <td>Cheema et al. (2022)</td>
            <td>Two emission measures</td>
            <td>Investor irrationality</td>
            <td>No</td>
        </tr>
        <tr>
            <td>Gorgen et al. (2020)</td>
            <td>BG scores</td>
            <td>Transition risk proxies</td>
            <td>No</td>
        </tr>
        <tr>
            <td>In et al. (2017)</td>
            <td>Emission intensity</td>
            <td>Investor irrationality</td>
            <td>No</td>
        </tr>
        <tr>
            <td>Pastor et al. (2022)</td>
            <td>E-scores</td>
            <td>Pastor & Stambaugh (2021)</td>
            <td>No</td>
        </tr>
    </tbody>
</table>
<p><strong>Summary of existing literature on climate risk:</strong> The table summarizes the literature relative to climate risk, with focus on the question about it being priced or not.</p>
"""

st.markdown(html_table, unsafe_allow_html=True)

st.write("""

Pastor et al. (2021, 2022) and Ardia et al. (2023)
provide a source of interpretation for these inconclusive results: 
         realized returns of transition risks 
         factor portfolios may strongly 
         differ from expected returns 
         due to unexpected change 
         in climate concerns. 
         In an event of unexpected 
         increase (decrease) 
         in climate concerns, 
         green (brown) stocks 
         may outperform, 
         resulting in positive (negative) 
         realized returns for a
        green-minus-brown factor for example. 
         
In that context, Pastor et al. (2022) have shown 
that a green factor portfolio outperforms in period 
of unexpected increase in climate concerns.
In this course, we are going to explore the transitory 
outperformance of the green factor.
         """)


st.write(
    """
In a first section, we build upon Apel et al. (2023) to evaluate the relationship 
between investable green investment portfolios (used as a proxy for the green factor) 
and transition risks concerns. 
"""
)

st.write(
    """
In a second section, we investigate the notion of "transition regimes": while individual increase 
or decrease in transition concerns may not be predictable, we may be able to identify periods
of increasing or decreasing transition concerns. We do so with regime-switching models.
""")

st.write(
    """
    In a final section, 
    we will explore the benefits of an active strategy that makes use of the information
    on transition regimes.
    """
)

# References section
st.write("### References")
st.write(
    """
    - Amenc, N., Goltz, F., Lodh, A., & Martellini, L. (2014). Towards Smart Equity Factor Indices: Harvesting Risk Premia without Taking Unrewarded Risks. *Journal of Portfolio Management*, 40(4), 106-122.
    - Apel, M., Betzer, A., & Scherer, B. (2023). Real-time transition risk. *Finance Research Letters*, 53, 103600.
    - Ardia, D., Bluteau, K., & Rüede, M. (2023). Climate Change Concerns and Investor Behavior. *Journal of Empirical Finance*, 68, 1-14.
    - Bolton, P., Despres, M., Pereira Da Silva, L. A., Samama, F., & Svartzman, R. (2020). The Green Swan: Central Banking and Financial Stability in the Age of Climate Change. *Bank for International Settlements*.
    - Bolton, P., Kacperczyk, M., & Samama, F. (2021). Do Investors Care about Carbon Risk? *Journal of Financial Economics*, 142(2), 517-549.
    - Bolton, P., Pereira Da Silva, L. A., Samama, F., & Svartzman, R. (2023). Climate Risk and Financial Stability. *Economic Policy Review*, Forthcoming.
    - Cheema, M. A., Ozturk, B., & Soytas, U. (2022). Climate Change and Stock Returns: Evidence from the Global Economy. *Energy Economics*, 104, 105699.
    - Gorgen, M., Jacob, A., Nerlinger, M., Riordan, R., Rohleder, M., & Wilkens, M. (2020). Carbon Risk. *European Financial Management*, 26(5), 1237-1273.
    - Hsu, P.-H., Li, K., & Tsou, C.-Y. (2023). Pollution and Performance: Industry Emissions and Financial Analysts. *Management Science*, 69(4), 1782-1801.
    - In, S. Y., Park, K. Y., & Monk, A. H. (2017). Being Green in the Credit Market. *Journal of Banking & Finance*, 77, 1-14.
    - Pastor, L., Stambaugh, R. F., & Taylor, L. A. (2022). Dissecting Green Returns. *Journal of Financial Economics*, 146(2), 403-424.
    - Pastor, L., & Stambaugh, R. F. (2021). Sustainable Investing in Equilibrium. *Journal of Financial Economics*, 142(2), 550-571.
    """
)