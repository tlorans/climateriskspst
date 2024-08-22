import streamlit as st
from pydantic import BaseModel, Field, conlist, field_validator 
from typing import List
import io
import contextlib

def run_code_and_capture_output(code: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            exec(code, globals(), locals())
        except Exception as e:
            print(e)
    return buffer.getvalue()

st.set_option('deprecation.showPyplotGlobalUse', False)


# Title of the web app
st.markdown("""
            # Testing the CAPM

            To check empirically the validity of the theory we have discussed, the first thing to test is the validity of the CAPM, which forms the basis of the theory.            
            As we have seen, cross-sectional variation in expected returns should be explained by the covariance between the excess return 
            of the asset and the excess return of the market portfolio. 
            The regression coefficient of this relationship is the market beta of the asset.
            In this part, we will first start by estimating the beta of the assets, and then apply portfolio univariate sort to test the CAPM.
            
            """)

st.markdown("""
## Estimating the CAPM Beta
            """)


st.markdown("""
## Univariate Portfolio Sort
            """)