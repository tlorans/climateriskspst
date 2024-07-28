import streamlit as st
from pydantic import BaseModel, Field, conlist, field_validator 
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

# Explanatory Text
st.markdown("""
### Objective of Pydantic

Pydantic is a data validation and settings management library for Python. It uses Python type annotations to provide a powerful and flexible way to define and validate data models. Pydantic ensures that data structures adhere to defined types and constraints, which is particularly useful for applications requiring rigorous data validation, such as financial modeling.

### Why Use Pydantic?

1. **Data Validation**: Pydantic automatically validates data against the types and constraints you specify. This reduces errors and ensures data integrity.
2. **Type Safety**: By leveraging Python's type annotations, Pydantic provides type safety, making your code more predictable and easier to debug.
3. **Declarative Syntax**: Pydantic's declarative syntax makes it easy to define complex data models with minimal code.
4. **Ease of Use**: Pydantic integrates seamlessly with other Python libraries and frameworks, making it a versatile choice for data validation.

### Example: Modeling a Stock with Pydantic

Let's dive into an example to see how Pydantic can be used to model financial data. We'll define a `Stock` model that represents a financial stock, including fields for the stock's name, symbol, weight, and carbon intensity.
""")

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

st.subheader("Example 1: A Valid Stock Model")
st.code(code_snippet_1, language='python')

if st.button('Run Example 1'):
    output = run_code_and_capture_output(code_snippet_1)
    st.subheader("Output")
    st.code(output, language='text')

# Explanatory Text for Example 2
st.markdown("""
### Handling Invalid Data

Pydantic not only helps in creating models with valid data but also in handling invalid data gracefully. 
Let's look at an example where invalid data is provided to the `Stock` model.
""")

# Example code snippet 2
code_snippet_2 = '''
from pydantic import BaseModel, Field, conlist
from typing import List

class Stock(BaseModel):
    name: str
    symbol: str
    weight: float = Field(..., ge=0, le=1)  # weight should be between 0 and 1
    carbon_intensity: float = Field(..., ge=0)  # carbon intensity should be non-negative

try:
    stock = Stock(name=123, symbol='AAPL', weight='no data', carbon_intensity=0.5)
    print(stock)
except Exception as e:
    print(e)
'''
st.subheader("Example 2: Unvalid Stock Model")
st.code(code_snippet_2, language='python')

if st.button('Run Example 2'):
    output = run_code_and_capture_output(code_snippet_2)
    st.subheader("Output")
    st.code(output, language='text')


class Stock(BaseModel):
    name: str
    symbol: str
    weight: float = Field(..., ge=0, le=1)  # weight should be between 0 and 1
    carbon_intensity: float = Field(..., ge=0)  # carbon intensity should be non-negative

# Explanatory Text for Example 3
st.markdown("""
### Creating a Portfolio Model

In finance, it's common to group multiple stocks into a portfolio. Pydantic makes it easy to model complex nested data structures like a portfolio containing multiple stocks. In this example, we'll create a `Portfolio` model that contains a list of `Stock` models.
""")
# Example code snippet 3

code_snippet_3 = '''
from pydantic import BaseModel, Field
from typing import List

class Stock(BaseModel):
    name: str
    symbol: str
    weight: float = Field(..., ge=0, le=1)  # weight should be between 0 and 1
    carbon_intensity: float = Field(..., ge=0)  # carbon intensity should be non-negative

class Portfolio(BaseModel):
    stocks: List[Stock]

try:
    stock1 = Stock(name='Apple Inc.', symbol='AAPL', weight=0.5, carbon_intensity=0.5)
    stock2 = Stock(name='Microsoft Corporation', symbol='MSFT', weight=0.3, carbon_intensity=0.3)
    stock3 = Stock(name='Amazon.com Inc.', symbol='AMZN', weight=0.2, carbon_intensity=0.2)
    portfolio = Portfolio(stocks=[stock1.dict(), stock2.dict(), stock3.dict()])  # Convert Stock instances to dictionaries
    print(portfolio)
except Exception as e:
    print(e)
'''

st.subheader("Example 3: Portfolio Model")
st.code(code_snippet_3, language='python')

if st.button('Run Example 3'):
    output = run_code_and_capture_output(code_snippet_3)
    st.subheader("Output")
    st.code(output, language='text')


# Explanatory Text for Example 4
st.markdown("""
### Adding Field Validators

In addition to basic type validation, 
Pydantic allows you to add custom validation logic to your models using field validators. 
Field validators can enforce more complex rules and dependencies between fields. 
In this example, we'll extend our `Portfolio` model to include a validator that ensures the 
portfolio contains at least one stock and that the total weight of all stocks sums to 1.
""")

code_snippet_4 = '''
from pydantic import BaseModel, Field, field_validator
from typing import List

class Stock(BaseModel):
    name: str
    symbol: str
    weight: float = Field(..., ge=0, le=1)  # weight should be between 0 and 1
    carbon_intensity: float = Field(..., ge=0)  # carbon intensity should be non-negative

class Portfolio(BaseModel):
    stocks: List[Stock]

    @field_validator('stocks')
    def check_stocks(cls, v):
        if len(v) < 1:
            raise ValueError('Portfolio must contain at least one stock')
        total_weight = sum(stock.weight for stock in v)
        if not (total_weight == 1.0):
            raise ValueError('Total weight of all stocks must sum to 1')
        return v

try:
    stock1 = Stock(name='Apple Inc.', symbol='AAPL', weight=0.5, carbon_intensity=0.5)
    stock2 = Stock(name='Microsoft Corporation', symbol='MSFT', weight=0.3, carbon_intensity=0.3)
    stock3 = Stock(name='Amazon.com Inc.', symbol='AMZN', weight=0.5, carbon_intensity=0.2)
    portfolio = Portfolio(stocks=[stock1.dict(), stock2.dict(), stock3.dict()])  # Convert Stock instances to dictionaries
    print(portfolio)
except Exception as e:
    print(e)
'''

st.subheader("Example 4: Field Validators")
st.code(code_snippet_4, language='python')

if st.button('Run Example 4'):
    output = run_code_and_capture_output(code_snippet_4)
    st.subheader("Output")
    st.code(output, language='text')

# Example code snippet 5

st.markdown("""
### Adding Computed Properties

Pydantic models can include computed properties using Python's `@property` decorator. This allows you to add methods that compute values based on the model's fields. In this example, we'll extend our `Portfolio` model to include a computed property that calculates the weighted average carbon intensity of the stocks in the portfolio.
""")

code_snippet_5 = '''
from pydantic import BaseModel, Field, field_validator
from typing import List

class Stock(BaseModel):
    name: str
    symbol: str
    weight: float = Field(..., ge=0, le=1)  # weight should be between 0 and 1
    carbon_intensity: float = Field(..., ge=0)  # carbon intensity should be non-negative

class Portfolio(BaseModel):
    stocks: List[Stock]

    @field_validator('stocks')
    def check_stocks(cls, v):
        if len(v) < 1:
            raise ValueError('Portfolio must contain at least one stock')
        total_weight = sum(stock.weight for stock in v)
        if not (total_weight == 1.0):
            raise ValueError('Total weight of all stocks must sum to 1')
        return v

        
    @property
    def weighted_average_carbon_intensity(self) -> float:
        weighted_sum = sum(stock.weight * stock.carbon_intensity for stock in self.stocks)
        return weighted_sum
try:
    stock1 = Stock(name='Apple Inc.', symbol='AAPL', weight=0.5, carbon_intensity=0.5)
    stock2 = Stock(name='Microsoft Corporation', symbol='MSFT', weight=0.3, carbon_intensity=0.3)
    stock3 = Stock(name='Amazon.com Inc.', symbol='AMZN', weight=0.2, carbon_intensity=0.2)
    portfolio = Portfolio(stocks=[stock1.dict(), stock2.dict(), stock3.dict()])  # Convert Stock instances to dictionaries
    print(portfolio)
except Exception as e:
    print(e)

print(f"Weighted Average Carbon Intensity: {portfolio.weighted_average_carbon_intensity}")
'''

st.subheader("Example 5: Property Decorator")
st.code(code_snippet_5, language='python')

if st.button('Run Example 5'):
    output = run_code_and_capture_output(code_snippet_5)
    st.subheader("Output")
    st.code(output, language='text')