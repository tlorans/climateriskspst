import streamlit as st
from pydantic import BaseModel, Field  # Ensure these are imported at the beginning
from typing import List
import io
import contextlib

# Title of the web app
st.title("Modelling Data with Pydantic")

def run_code_and_capture_output(code: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            exec(code, globals(), locals())
        except Exception as e:
            print(e)
    return buffer.getvalue()

# Example code snippet 1
code_snippet_1 = '''
from pydantic import BaseModel, Field, conlist
from typing import List

class Stock(BaseModel):
    name: str
    symbol: str
    weight: float = Field(..., ge=0, le=1)  # weight should be between 0 and 1
    carbon_intensity: float = Field(..., ge=0)  # carbon intensity should be non-negative

try:
    stock = Stock(name='Apple Inc.', symbol='AAPL', weight=0.5, carbon_intensity=0.5)
    print(stock)
except Exception as e:
    print(e)
'''

st.subheader("Example 1: Stock Model")
st.code(code_snippet_1, language='python')

if st.button('Run Example 1'):
    output = run_code_and_capture_output(code_snippet_1)
    st.subheader("Output")
    st.code(output, language='text')
