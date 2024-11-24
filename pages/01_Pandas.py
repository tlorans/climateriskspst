import pandas as pd
import streamlit as st

st.title("Introduction to Pandas")

st.write(
    """
    Pandas is a powerful data manipulation library for Python. 
    It provides data structures like Series and DataFrame
    that make working with data easier.
    """
)

st.write("""To use pandas, you first need to install it using pip:""")

st.code("pip install pandas")

st.write(
    """
    Once you've installed pandas, you can import it using the following code:
    """
)

st.code("import pandas as pd")

st.write("Here's a simple DataFrame:")

st.code('''
# Simple DataFrame example
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"],
}
df = pd.DataFrame(data)
        ''', language='python')

# Simple DataFrame example
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"],
}
df = pd.DataFrame(data)

st.dataframe(df)

st.write("A simple way to get the data based on specific conditions is:")

st.code('''
print(df.query('Age > 30'))
        ''', language='python')

st.write("The output will be:")

st.dataframe(df.query('Age > 30'))


st.subheader('Loading Data')

st.subheader('Exploring Data')

st.subheader('Manipulating DataFrames')

st.subheader('Time Series')