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

st.write("""
You can load data from a variety of sources, including CSV files, Excel files, and SQL databases.
Here's an example of loading data from an Excel file, using the `read_excel` function:
         """)

st.code('''
data = pd.read_excel('data/tri.xlsx', sheet_name="weekly")
        ''', language='python')


data = (
    pd.read_excel('data/tri.xlsx', sheet_name="weekly")
)

st.dataframe(data)

st.write("""
For csv files, you can use the `read_csv` function. We'll see later 
how to use `pandas_datareader` to load data from internet sources. Especially for financial data.
         """)

st.subheader('Exploring Data')

st.write("""
         As a first step in exploring your data, you can use the `head` and `tail` methods to view the first and last few rows of your DataFrame.
            """)    

st.code('''
        print(data.head())
        ''', language='python')

st.dataframe(data.head())

st.write("""You may also want to use the `describe` method to get a summary of the numerical columns in your DataFrame.""")

st.code('''
print(data.describe())
        ''', language='python')

st.dataframe(data.describe())

st.subheader('Manipulating DataFrames')

st.write("""
Again, we can use the `query` method to filter the data based on specific conditions.
         """)


st.code('''
print(data.query('Date > "2020-01-01"'))
        ''', language='python')

st.dataframe(data.query('Date > "2020-01-01"'))

st.write("""
         We can rename columns using the `rename` method:
            """)

st.code('''
print(data.rename(columns={"Date": "date"}))
        ''', language='python')

st.dataframe(data.rename(columns={"Date": "date"}))

st.write("""We can add columns using the `assign` method:""")

st.code('''
print(data.assign(cum_tri = lambda x: x['TRI_innovation_weekly'].rolling(4).mean()))
        ''', language='python')

st.dataframe(data.assign(cum_tri = lambda x: x['TRI_innovation_weekly'].rolling(4).mean()))  

st.write("""
         We can finally reshape the data using the `melt` method to convert it from wide to long format:
            """)

st.code('''
print(data.melt(id_vars=['Date'], value_vars=['TRI_innovation_weekly'], var_name='TRI', value_name='value'))
        ''', language='python')

st.dataframe(data.melt(id_vars=['Date'], value_vars=['TRI_innovation_weekly'], var_name='TRI', value_name='value'))

st.subheader('Pipeline')