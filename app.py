import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = None

PAGES = ["Theory", "Empirics", "Portfolio Construction"]

def set_page(page):
    st.session_state.page = page
    st.rerun()

def home_page():
    st.header("Welcome")
    st.write("Please choose a section to navigate:")
    if st.button("Theory"):
        set_page("Theory")
    if st.button("Empirics"):
        set_page("Empirics")
    if st.button("Portfolio Construction"):
        set_page("Portfolio Construction")

def go_home():
    st.session_state.page = None
    st.rerun()

page = st.session_state.page
go_home_page = st.Page(go_home, title="Home", icon="🏠", default=(page is None))

# Define pages
portfolio_page = [st.Page("pc/introduction.py", title="Introduction", default=(page=="Portfolio Construction")),
                  st.Page("pc/parametric_portfolio_policies.py", title="Parametric Portfolio Policies"),
]

empirics_pages = [
    st.Page("emp/introduction.py", title = "Introduction", default=(page == 'Empirics')),
    st.Page("emp/capm.py", title="Testing the CAPM"),
    st.Page("emp/famafrench.py", title="Fama-French Factors"),
    st.Page("emp/greenfactor.py", title="Green Factor"),
]

kc_pages = [
    st.Page("kc/introduction.py", title="Climate Risks and Equity Portfolio", default = (page == "Theory")),
    st.Page("kc/uncertainty.py", title="Climate Change Uncertainty"),
    st.Page("kc/optimalportfolio.py", title="Optimal Portfolio with Climate Change Uncertainty"),
    st.Page("kc/expectedreturn.py", title="Expected Return with Climate Change Uncertainty"),
    st.Page("kc/hedgingportfolio.py", title="Climate Risks Hedging Portfolio"),
    st.Page("kc/unexpectedreturns.py", title="Unexpected Returns: a Consequence of Resolving Climate Change Uncertainty"),

]

# Main navigation setup

st.title("Climate Risks")
# st.image("images/horizontal_blue.png")

page_dict = {}

if st.session_state.page == "Theory":
    page_dict["Theory"] = kc_pages
elif st.session_state.page == "Empirics":
    page_dict["Empirics"] = empirics_pages
elif st.session_state.page == "Portfolio Construction":
    page_dict["Portfolio Construction"] = [portfolio_page]

if len(page_dict) > 0:
    pg = st.navigation({"": [go_home_page]} | page_dict)
else:
    home_page()

# This line runs the selected page from the navigation, if any.
if 'pg' in locals():
    pg.run()
