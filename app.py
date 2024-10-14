import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = None

PAGES = ["Rewarded and Unrewarded Risks","What About Climate Risks?","Hedging Unrewarded Risks"]

def set_page(page):
    st.session_state.page = page
    st.rerun()

def home_page():
    st.header("Welcome")
    st.write("Please choose a section to navigate:")
    if st.button("Rewarded and Unrewarded Risks"):
        set_page("Rewarded and Unrewarded Risks")
    if st.button("What About Climate Risks?"):
        set_page("What About Climate Risks?")
    if st.button("Hedging Unrewarded Risks"):
        set_page("Hedging Unrewarded Risks")

def go_home():
    st.session_state.page = None
    st.rerun()

page = st.session_state.page
go_home_page = st.Page(go_home, title="Home", icon="🏠", default=(page is None))

# Define pages
rewarded_unrewarded = [st.Page("rewarded_unrewarded/rewarded.py", title = "Rewarded Risks", default=(page=="Rewarded and Unrewarded Risks")),
                st.Page("rewarded_unrewarded/unrewarded.py",title ="Unrewarded Risks"),
]

climate_risks = [st.Page("climate_risks/characteristics.py", title = "Climate Characteristics", default=(page=="What About Climate Risks?")),
                 st.Page("climate_risks/climate_lambda.py", title = "Where is the Risk Premium?"),
                # st.Page("climate_risks/unexpected_returns.py", title = "Unexpected Returns and Climate Concerns"),
]

hedging = [st.Page("hedging_unrewarded_risks/hedging_portfolio.py", title = "Hedging Portfolio", default=(page=="Hedging Unrewarded Risks")),
        #    st.Page("hedging_unrewarded_risks/hedging_portfolio.py", title = "Hedging Portfolio"),
]


# Main navigation setup
st.title("Climate Risks")

page_dict = {}

if st.session_state.page == "Rewarded and Unrewarded Risks":
    page_dict["Rewarded and Unrewarded Risks"] = rewarded_unrewarded
elif st.session_state.page == "What About Climate Risks?":
    page_dict["What About Climate Risks?"] = climate_risks
elif st.session_state.page == "Hedging Unrewarded Risks":
    page_dict["Hedging Unrewarded Risks"] = hedging


if len(page_dict) > 0:
    pg = st.navigation({"": [go_home_page]} | page_dict)
else:
    home_page()

# This line runs the selected page from the navigation, if any.
if 'pg' in locals():
    pg.run()
