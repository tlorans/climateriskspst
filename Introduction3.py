import streamlit as st
import sympy as sp
import numpy as np

st.title('Introduction to Bayesian Portfolio Optimization')

n = 3

# Diagonal matrix for view portfolios (one per factor)
P = sp.Matrix([[0,0,0],[-1, 1, 0], [-1, 0, 1]]) # view portfolios, k views and n securities (factor + market)

st.latex('P = ' + sp.latex(P))  

# expected returns of the view portfolios
v = sp.Matrix([0, 0.03, 0.04]) # expected returns of the view portfolios

st.latex('v = ' + sp.latex(v))

# w^bmk the benchmark weights
w_bmk = sp.Matrix([1/n]*n)

st.latex('w^{bmk} = ' + sp.latex(w_bmk))

delta = 2.5 # risk aversion
Sigma = sp.diag(0.2**2, 0.15**2, 0.1**2) # prior covariance matrix

st.latex('\Sigma = ' + sp.latex(Sigma))
tau = 0.025 # parameter in the views and the prior
Omega = sp.diag(0.0001, 0.0001, 0.0001) # uncertainty in the views

pi = delta * Sigma * w_bmk # equilibrium excess returns

st.latex('\pi = ' + sp.latex(pi))

# view portfolio weights
lambda_ = delta**(-1) * (P * Sigma * P.T + Omega / tau) ** -1 * (v - P * pi)

st.latex('\lambda = ' + sp.latex(lambda_))

active_weights = P.T * lambda_

st.latex('w^{act} = ' + sp.latex(active_weights))

# litterman weights
w_lit = w_bmk + active_weights

st.latex('w^{lit} = ' + sp.latex(w_lit))