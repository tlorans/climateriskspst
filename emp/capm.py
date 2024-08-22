import streamlit as st
from pydantic import BaseModel, Field, conlist, field_validator 
from typing import List
import io
import contextlib

# Title of the web app
st.title("Testing the CAPM Beta")


st.markdown("""
## Estimating the CAPM Beta
            """)


st.markdown("""
## Univariate Portfolio Sort
            """)