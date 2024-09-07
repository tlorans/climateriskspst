import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = None

PAGES = ["Risk", "Hedging"]

def set_page(page):
    st.session_state.page = page
    st.rerun()

def home_page():
    st.header("Welcome")
    st.write("Please choose a section to navigate:")
    if st.button("Risk"):
        set_page("Risk")
    if st.button("Hedging"):
        set_page("Hedging")

def go_home():
    st.session_state.page = None
    st.rerun()

page = st.session_state.page
go_home_page = st.Page(go_home, title="Home", icon="🏠", default=(page is None))

# Define pages
risk_pages = [st.Page("risk/introduction.py", title="Introduction", default=(page=="Risk")),
                st.Page("risk/risk_factor.py", title="Risk Factor"),
                st.Page("risk/climate_risk_factor.py", title="Climate Risk Factor"),
                st.Page("risk/unpriced_risk.py", title="Unpriced Risk"),
                st.Page("risk/sources_risk.py", title="Sources of Risk"),
]

hedging_pages = [
    st.Page("hedging/introduction.py", title = "Introduction", default=(page == 'Empirics')),
]


# Main navigation setup

st.title("Climate Risks")
# st.image("images/horizontal_blue.png")

page_dict = {}

if st.session_state.page == "Risk":
    page_dict["Risk"] = risk_pages
elif st.session_state.page == "Empirics":
    page_dict["Empirics"] = hedging_pages


if len(page_dict) > 0:
    pg = st.navigation({"": [go_home_page]} | page_dict)
else:
    home_page()

# This line runs the selected page from the navigation, if any.
if 'pg' in locals():
    pg.run()
