import streamlit as st 

st.markdown("""
# Climate Risks

*Climate risks* are risks that changes in climate will negatively impact cash flows in the future (De Nard *et al.*, 2024 [^1]).
Two broad categories are mainly considered in the literature:
- **Physical climate risk:** Includes risks of the direct impairment of productive assets due to climate change, such as floods, hurricanes, or droughts.
- **Transition risk:** Includes risks to cash flows and asset values due to *climate mitigation policies* (Giglio *et al.* 2021 [^2]).
  
  > Climate mitigation policies are policies that aim to reduce greenhouse gas emissions, such as carbon taxes, cap-and-trade systems, or renewable energy subsidies.

Both are sources of uncertainty. We do not know the exact future evolution of the climate, nor the exact future evolution of climate policies.

## Climate Risks Hedging

Different stocks may be positively or negatively exposed to climate risks. It means that unanticipated climate shocks may have "winners" or "losers" in the stock market. For example, companies that are heavily dependent on fossil fuels may be negatively exposed to change in mitigation policies, whereas companies that are developing renewable energies may be positively exposed. For investors, hedging climate risks would involve longing stocks that do well in the period of realization of climate shocks and shorting stocks that do poorly in this period.

## Strong Uncertainty in Climate Change

This definition of climate risks as the uncertainty surrounding the underlying dynamics is in line with the traditional risk definition in asset pricing. Indeed, as mentioned by Giglio *et al.* (2021) [^2], asset pricing models and resulting risk management strategies are usually formulated under the assumption of rational expectations, so that investors inside the model know the exact probability laws that govern their environment. In rational expectations models, investors face uncertainty about stochastic shocks from a *known* probability distribution. But this may not be the case for climate risks. Indeed, it seems implausible that investors know the precise distribution or severity of climate risks that are facing them. As Lemoine (2020) notes:

> "Uncertainty is fundamental to climate change. Today's greenhouse gas emissions will affect the climate for centuries. The emission price that internalizes the resulting damages depends on the uncertain degree to which emissions generate warming, on the uncertain channels through which warming will impact consumption and the environment, on the uncertain future evolution of greenhouse gas stocks, and on uncertain future growth in productivity and consumption."

As noted by Brock and Hansen (2017) [^3], the climate change complexity makes it especially challenging to model, and adds ambiguity and model misspecification to the usual risk component of uncertainty.

## Pastor-Stambaugh-Taylor (2021) Model

The Pastor-Stambaugh-Taylor (2021) model [^4] is a first step to understand the implications of climate risks for investors and the strong uncertainty that comes with it. In a simple CAPM framework, investors have heterogeneous beliefs about climate risks, reflecting the uncertainty about the future evolution of the climate. Investors form portfolios that reflect their beliefs about climate risks. The CAPM alpha (i.e., the resulting climate risks premium) depends on the average climate risks perception of investors. If the average perception of climate risks is low, the climate risks premium is insignificant. The climate risks hedging portfolio is proportional to the stocks' sensitivity to climate risks. The higher the investor's perception of climate risks compared to the average, the more the investor will short the climate risks mimicking portfolio (i.e., will short the stocks that are exposed to climate risks). Investors with average perception of climate risks will hold the market portfolio only.

In the following, we investigate how investors can manage their exposure to climate risks, in the presence of strong uncertainty about the future evolution of the climate. In the first part, we present the components of climate uncertainty, highlighting the challenge in dealing with climate risks. We then show how investors with different perceptions of climate risks can form their optimal portfolios. The combined perceptions of climate risks of all investors determine the level of the climate risks premium. We then show how we can reexpress the optimal portfolio weights in terms of allocation between the market portfolio and the climate risks hedging portfolio, with the latter being proportional to the stocks' sensitivity to climate risks, and the allocation being determined by the investor's perception of climate risks. We also show how unanticipated changes in climate risks perception may incur unexpected returns. Finally, we propose two simple strategies as an illustration of practical implications of the PST model.

**Main practical implications:**
- Due to the strong uncertainty of climate risks, investors may have different perceptions of climate risks, leading to different optimal portfolios.
- Stocks' sensitivities to climate risks are the main ingredients to form the climate risks hedging portfolio, and should therefore be the main priority for building solutions for investors.

[^1]: De Nard et al., 2024, "Factor Model for Climate Risks."
[^2]: Giglio et al., 2021, "Climate Mitigation Policies and Economic Implications."
[^3]: Brock and Hansen, 2017, "Modeling and Pricing Climate Change Risks."
[^4]: Pastor-Stambaugh-Taylor, 2021, "Sustainable Asset Pricing Models."
""")

