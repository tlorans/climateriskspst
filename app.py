import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = None

PAGES = ["Investment Philosophy", "Investment Strategy"]

def set_page(page):
    st.session_state.page = page
    st.rerun()

def home_page():
    st.header("Welcome")
    st.write("Please choose a section to navigate:")
    if st.button("Investment Philosophy"):
        set_page("Investment Philosophy")
    if st.button("Investment Strategy"):
        set_page("Investment Strategy")

def go_home():
    st.session_state.page = None
    st.rerun()

page = st.session_state.page
go_home_page = st.Page(go_home, title="Home", icon="🏠", default=(page is None))

# Define pages
portfolio_page = [st.Page("pc/portfolio.py", title="Portfolio", default=(page=="Investment Strategy")),
                  st.Page("pc/practical1.py", title="Practical Portfolio 1: Manage My Sensitivity!"),
                    # st.Page("pc/practical2.py", title="Practical Implication 2: Make My Portfolio Great Again!"),
]


kc_pages = [
    st.Page("kc/introduction.py", title="Climate Risks and Equity Portfolio", default = (page == "Investment Philosophy")),
    st.Page("kc/optimalportfolio.py", title="Optimal Portfolio with Climate Change Uncertainty"),
    st.Page("kc/expectedreturn.py", title="Expected Return with Climate Change Uncertainty"),
    st.Page("kc/hedgingportfolio.py", title="Climate Risks Hedging Portfolio"),

]

# Main navigation setup

st.title("Climate Risks")
# st.image("images/horizontal_blue.png")

page_dict = {}

if st.session_state.page == "Investment Philosophy":
    page_dict["Investment Philosophy"] = kc_pages
elif st.session_state.page == "Investment Strategy":
    page_dict["Investment Strategy"] = portfolio_page

if len(page_dict) > 0:
    pg = st.navigation({"": [go_home_page]} | page_dict)
else:
    home_page()

# This line runs the selected page from the navigation, if any.
if 'pg' in locals():
    pg.run()
