import streamlit as st
from pydantic import BaseModel, Field, conlist, field_validator 
from typing import List
import io
import contextlib
import pandas as pd

def run_code_and_capture_output(code: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            exec(code, globals(), locals())
        except Exception as e:
            print(e)
    return buffer.getvalue()

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("""# Green Factor""")