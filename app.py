import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = None

PAGES = ["Knowledge Center", "Portfolio Construction"]

def set_page(page):
    st.session_state.page = page
    st.rerun()

def home_page():
    st.header("Welcome")
    st.write("Please choose a section to navigate:")
    if st.button("Knowledge Center"):
        set_page("Knowledge Center")
    if st.button("Portfolio Construction"):
        set_page("Portfolio Construction")

def go_home():
    st.session_state.page = None
    st.rerun()

page = st.session_state.page
go_home_page = st.Page(go_home, title="Home", icon="🏠", default=(page is None))

# Define pages
pc_pages = [
    st.Page("pc/data_model.py", title="Modelling Data with Pydantic", default=(page=="Portfolio Construction")),
]

kc_pages = [
    st.Page("kc/introduction.py", title="Climate Risks and Equity Portfolio", default = (page == "Knowledge Center")),
    st.Page("kc/uncertainty.py", title="Climate Change Uncertainty"),
    st.Page("kc/optimalportfolio.py", title="Optimal Portfolio with Climate Change Uncertainty"),
    st.Page("kc/expectedreturn.py", title="Expected Return with Climate Change Uncertainty"),
    st.Page("kc/hedgingportfolio.py", title="Climate Risks Hedging Portfolio"),
    st.Page("kc/unexpectedreturns.py", title="Unexpected Returns: a Consequence of Resolving Climate Change Uncertainty"),
    st.Page("kc/practical1.py", title="Practical Portfolio 1: Manage My Sensitivity!"),
    st.Page("kc/practical2.py", title="Practical Implication 2: Make My Portfolio Great Again!"),
    st.Page("kc/climate_betas.py", title="Climate Betas")
]

# Main navigation setup

st.title("Climate Risks")
# st.image("images/horizontal_blue.png")

page_dict = {}

if st.session_state.page == "Knowledge Center":
    page_dict["Knowledge Center"] = kc_pages
elif st.session_state.page == "Portfolio Construction":
    page_dict["Portfolio Construction"] = pc_pages

if len(page_dict) > 0:
    pg = st.navigation({"": [go_home_page]} | page_dict)
else:
    home_page()

# This line runs the selected page from the navigation, if any.
if 'pg' in locals():
    pg.run()
