import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = None

PAGES = ["Rewarded and Unreward Risks","Hedging Unrewarded Risks"]

def set_page(page):
    st.session_state.page = page
    st.rerun()

def home_page():
    st.header("Welcome")
    st.write("Please choose a section to navigate:")
    if st.button("Rewarded and Unrewarded Risks"):
        set_page("Rewarded and Unrewarded Risks")
    if st.button("Hedging Unrewarded Risks"):
        set_page("Hedging Unrewarded Risks")

def go_home():
    st.session_state.page = None
    st.rerun()

page = st.session_state.page
go_home_page = st.Page(go_home, title="Home", icon="🏠", default=(page is None))

# Define pages
first_part = [st.Page("rewarded_unrewarded/what_are_climate_risks.py", title = "Climate Risks and Stock Returns", default=(page=="Rewarded and Unrewarded Risks")),
                st.Page("rewarded_unrewarded/rewarded_unrewarded.py",title ="Rewarded and Unrewarded Risks"),
                    st.Page("rewarded_unrewarded/project_1.py", title = "Project 1"),
                st.Page("rewarded_unrewarded/climate_risks.py", title = "Climate Risks"),
              st.Page("rewarded_unrewarded/project_2.py", title = "Project 2"),
]

second_part = [st.Page("hedging_unrewarded_risks/targeted.py", title = "Hedging Portfolio", default=(page=="Hedging Unrewarded Risks")),
               st.Page("hedging_unrewarded_risks/project.py", title = "Project 3"),
]


# Main navigation setup
st.title("Climate Risks")

page_dict = {}

if st.session_state.page == "Rewarded and Unrewarded Risks":
    page_dict["Rewarded and Unrewarded Risks"] = first_part
elif st.session_state.page == "Hedging Unrewarded Risks":
    page_dict["Hedging Unrewarded Risks"] = second_part


if len(page_dict) > 0:
    pg = st.navigation({"": [go_home_page]} | page_dict)
else:
    home_page()

# This line runs the selected page from the navigation, if any.
if 'pg' in locals():
    pg.run()
