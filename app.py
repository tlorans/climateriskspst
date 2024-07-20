# app.py
import streamlit as st
# from sections.home import home, portfolio_construction
# from sections.kc import (
#     introduction, uncertainty, optimalportfolio, expectedreturn,
#     hedgingportfolio, unexpectedreturns, practical1, practical2, climate_betas
# )



# Define pages
home_page = st.Page("sections/home/home.py", title="Home")
knowledge_center_page = st.Page("sections/home/knowledge_center.py", title="Knowledge Center")
portfolio_page = st.Page("sections/home/portfolio_construction.py", title="Portfolio Construction")

kc_pages = [
    st.Page("sections/kc/introduction.py", title="Climate Risks and Equity Portfolio"),
    st.Page("sections/kc/uncertainty.py", title="Climate Change Uncertainty"),
    st.Page("sections/kc/optimalportfolio.py", title="Optimal Portfolio with Climate Change Uncertainty"),
    st.Page("sections/kc/expectedreturn.py", title="Expected Return with Climate Change Uncertainty"),
    st.Page("sections/kc/hedgingportfolio.py", title="Climate Risks Hedging Portfolio"),
    st.Page("sections/kc/unexpectedreturns.py", title="Unexpected Returns: a Consequence of Resolving Climate Change Uncertainty"),
    st.Page("sections/kc/practical1.py", title="Practical Portfolio 1: Manage My Sensitivity!"),
    st.Page("sections/kc/practical2.py", title="Practical Implication 2: Make My Portfolio Great Again!"),
    st.Page("sections/kc/climate_betas.py", title="Climate Betas")
]

# Main navigation setup
st.set_page_config(page_title="Climate Risks")

pages = {
    "Knowledge Center" : kc_pages
}

pg = st.navigation(pages)
pg.run()


