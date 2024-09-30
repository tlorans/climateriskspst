import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
import sympy as sp
import sympy.stats as stats
import plotly.graph_objs as go


st.title('Climate Risks and the Cross Section of Stock Returns')


st.write(r"""
Two main currents characterise the literatuer 
on climate finance. The first relates 
to climate change science and the related risks. 
The second describes the financial implications
         for asset pricing.

Climate change risks can be divided into physical and transition risks 
(Carney, 2015). Physical risks refer to the negative impact 
of climate and weather-related events on company operations or society 
         (Tankov and Tantet, 2019). 
We traditionaly think of these risks as two types:
         (i) acute physical risks, that are related 
         to extreme weather events, and (ii) chronic physical risks,
         that are related to long-term shifts in climate patterns or sea-level rise.

On the other hand, transition risks refer to all 
         possible scenarios coherent with a path to a low-carbon 
         economy and all related implications for fossil fuels 
         and dependent sectors (Curtin $\textit{et al.}$, 2019).

In this course, we will focus on transition risks,
and discuss how their dynamics matter for asset pricing.
         """)

st.write(r"""
With respect to the financial side, climate finance 
         literature can be divided according to the type 
         of financial risks and asset considered.
In this course, we will focus on the impact on equity.
         """)

st.write(r"""
         In this section, 
we will discuss how firm value can be affected 
by either the physical or the transition risks 
         channels. 
            """)

st.subheader('Physical Risks Channel')

st.write(r"""
At the more fundamental level, physical risks 
depend on the following three drivers (Tankov and Tantet, 2019):
         """)

st.latex(r'''
         \begin{equation}
         \text{Physical Risk} = f(\text{Climate Hazard} \times \text{Exposure} \times \text{Vulnerability})
         \end{equation}
         ''')

st.write(r"""
    The variable $\textit{Climate Hazard}$ refers to the climate events 
         or weather patterns of interest, both interms of physical intensity 
         and in probability of occurence. The term $\textit{Exposure}$
         represents the geographical distribution of the entity 
         that the climate hazard might impact, and 
            $\textit{Vulnerability}$ represents the threats 
         to the asset from of its exposure to the climate hazard.
         Each term in the first equation requires 
         specific and geo-spatial data to adapt the analysis to the particular 
         entity of interest.
            """)

st.write(r"""
         The term $\textit{Climate Hazard}$ can be analysed 
         using two different types of datasets, 
         namely climate and natural disaster datasets. 
         Regarding the former, Tankov and Tantet (2019)
         classified four different types of climate 
         datasets for modelling physical risks. This 
         classification ranks climate datasets according to the
         levels of modelling required to create them,
         distinguishing between: (i) observational datasets,
         (ii) reanalysis datasets, (iii) projections 
         datasets and (iv) climate indices. 
            """)

st.write(r"""
         Altough traditionally used in disaster risk management,
         natural disaster datasets are gaining momentum in climate finance.
         Natural hazards can be divided into: (i) hydrological;
         (ii) meteorological; (iii) climatological and 
         (iv) geophysical.
         Natural disaster datasets are attractive in climate finance because,
         unlike climate datasets, they provide both the location and 
         the historical economic damages of a certain physical hazard. 
         """)

st.write(r"""
         The quantitative assessment at the firm level of the
         two other variables in the first equation 
         requires the use of asset-level data. The latter represents 
         any type of quantitative or qualitative information 
         regarding physical assets, including their characteristics, 
         geographical locations or orwnerhsip.
         Moreover, the level of a firm's vulenrability to 
         physical risks is negatively related to its adaptation capacity.
         Thus, modelling this channel further requires data geared toward 
         firm innovation and resilience.
         """)


st.subheader('Transition Risks Channel')

st.write(r"""
The choice of a framework to catagorise transition risks 
         is less canonical than the one discussed for physical risks.
         Nevertheless, Semieniuk $\textit{et al.}$ (2021) provided 
         a taxonomy that allows one to identify several transition risks
         drivers that may lead to economic impacts in financial markets.
            """)

st.write(r"""
Transition risks are a combination of three factors (Semienuk $\textit{et al.}$, 2021):
            """)

st.latex(r'''
            \begin{equation}
            \text{Transition Risk} = f(\text{Policy Risk} \times \text{Technology Risk} \times \text{Preference Change})
            \end{equation}
            ''')

st.write(r"""
The term $\textit{Policy Risk}$ refers to the risks and 
         opportunities that may be triggered by climate 
         mitigation policies. The aim of these policies is to 
         reduce the amount of greenhouse gas (GHG) emissions in the atmosphere,
         especially carbon dioxide (CO2) emissions. The reason 
         for mitigation policies to focus on CO2 emissions 
         is that these kinds of emissions are considered the primary 
         factor in human-induced global warming (Nordhaus, 1977, 1993).
         Cliamte mitigation policies can be implemented via 
         market-based and non-market-based mechanisms.
        Market-based mechanisms comprise the two forms 
         of carbon pricing: (i) carbon taxes and (ii) cap-and-trade schemes.
         Non-market-based mechanisms are related to (i) environmental regulation;
         (ii) green subsidies and (iii) voluntary commitments by 
            firms or governments.
                """)

st.write(r"""
The Greenhouse Gas Protocol (GHG Protocol) sets the standards 
         for quantify corporate emissions at the firm level,
         distinguishing between three different sources of 
         GHG emissions. Scope 1 emissions refer to the direct emissions 
         from plants owned or controlled by a company.
         Scope 2 and scope 3 emissions represent two forms of indirect emissions.
         In particular, scope 2 emissions arise from the generation 
         of purchased steam, heat and electricity consumed by the firm,
         while scope 3 emissions come from sources not owned or controlled 
         by the company, such as emissions from outsourced activities. 
            """)    

st.write(r"""Studies in climate finance tend to assume 
         that the higher a company's cope emissions, the browner 
         that firm is (Bolton and Kacperczyk, 2021). The opposite holds 
         for green firms. Moreover, given these sources of scope emissions,
         three different measures at the plant or firm level can be derived.
         The first is the total level of emissions, that may be decomposed for each
         type of scope emission subcategory. Year-by-year change 
         in emissions quantifies the growth rate in annual corporate emissions.
         Finally, one could also compute an emission intensity measure,
         quantifying carbon emissions per unit of sales or assets.
         Each of these measures can be used to estimate the degree of exposure a company
         may have to different mitigation policies (Bolton and Kacperczyk, 2021).
            """)

st.write(r"""
The variable $\textit{Technology Risk}$ refers to the introduction 
         of cost-saving technologies that would forster
         the adoption of low-carbon energy sources. Notably, society is still
         reliant on fossil fuel energies. However, several factors may be pushing firms 
         to adopt low-carbon energy sources, such as expected liabilities toward 
         environmental policies, investor pressure or simply to maintain competition in the market.
         Thus modelling technology risk requires firm-level information about
         (i) innovation data; (ii) emission data. From the 
         motivations that could trigger firms to adopt carbon management practices, it is clear 
         that the variable $\textit{Technology Risk}$ and $\textit{Policy Risk}$ are closely related.
         """)

st.write(r"""
Finallt, the term $\textit{Preference Change}$ may be related to two
         non-mutually exclusive channels (Pastor $\textit{et al.}$ (2020)).
         The first refers to unexpected preference changes in 
         green-motivated consumers' tastes. This group's environmental concerns 
         may positively affect the cash flows of green firms.
         The second channel is represented by unexpected shifts 
         in investor preferences toward carbon-intensive assets. 
         Investors may change their preferences regarding these assets 
         for both pecuniary and nonpecuniary motives. In the former case,
         the financial logic is tied to risk-return considerations, as carbon assets 
         may be deemd to have a higher downside risk. On the other 
         hand, nonpecuniary motives are related to the ethical benefits 
         an investor derives when holding climate-friendly assets (Fama and French, 2007).
         Modelling the investor channel requires data about (i) investor survey or (ii)
         financial flows toward green labelled funds.
            """)


