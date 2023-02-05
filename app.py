import streamlit as st
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices# pip install PyPortfolioOpt
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import objective_functions
import os	# for os.path.join
import copy	# for deepcopy
import openai # pip install openai
import numpy as np
import pandas as pd
import yfinance as yf # pip install yfinance
import plotly.express as px	# pip install plotly
import matplotlib.pyplot as plt
import seaborn as sns	# pip install seaborn
from datetime import datetime
from io import BytesIO # for downloading files
import logging	# for logging
import pickle # pip install pickle5

# -------------- PAGE CONFIG --------------
page_title = "Financial Portfolio Optimizer"
page_icon = ":zap:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"

st.set_page_config(page_title = page_title, layout = layout, page_icon = page_icon)
st.title(page_title + " " + page_icon)

# Streamlit sidebar table of contents
st.sidebar.markdown('''
# Sections
- [Optimized Max Sharpe Portfolio Weights](#optimized-max-sharpe-portfolio-weights)
- [Performance Expectations](#performance-expectations)
- [Correlation Matrix](#correlation-matrix)
- [Individual Stock Prices and Cumulative Returns](#individual-stock-prices-and-cumulative-returns)
- [AI Generated Report](#summary)
''', unsafe_allow_html=True)

st.markdown('### 1. How Much?')
# 1. AMOUNT
# Enter investment amount and display it. It must be an integer not a string
amount = 100
st.markdown("""---""")

st.markdown('''### 2. How Long?
Enter start and end dates for backtesting your portfolio''') # TODO: Instead of entering start and end dates, have them enter 
# number of years the user plans on holding the portfolio - then have the app go back that many years for the backtesting
# 2.TIME HORIZON
col1, col2 = st.columns(2)  # split the screen into two columns. columns(2) says split the screen into two columns
							# if said columns(1,2) then the first column would be 1/3 of the screen and the second column would be 2/3 of the screen
with col1:
	start_date = st.date_input("Start Date",datetime(2020, 1, 1))
	
with col2:
	end_date = st.date_input("End Date") # it defaults to current date
st.markdown("""---""")

# 3. TICKERS
st.markdown('''### 3. What?
Enter assets you would like to test as a portfolio''')
tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas \
								WITHOUT spaces, e.g. "TSLA,AAPL,MSFT,ETH-USD,BTC-USD,MATIC-USD,GOOG"', 'TSLA,AAPL,MSFT,ETH-USD,BTC-USD,MATIC-USD,GOOG').upper()
tickers = tickers_string.split(',')
st.markdown("""---""")

risk_free_rate = .02
st.write('Risk Free Rate: ', risk_free_rate)
st.markdown("""---""")

# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# -------------- FUNCTIONS ----------------
# Plot cumulative returns
def plot_cum_returns(data, title):    
	daily_cum_returns = 1 + data.dropna().pct_change()
	daily_cum_returns = daily_cum_returns.cumprod()*100 ### is this 100 for percentage or does it represent the initial investment amount? Answer: 100 for percentage
	fig = px.line(daily_cum_returns, title=title)
	return fig
# Efficient frontier
def plot_efficient_frontier_and_max_sharpe(mu, S): # mu is expected returns, S is covariance matrix. So we are defining a function that takes in these two parameters
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S) # the efficient frontier object is 
	fig, ax = plt.subplots(figsize=(6,4)) # fig, ax = plt.subplots() is the same as fig = plt.figure() and ax = fig.add_subplot(111)
	ef_max_sharpe = pickle.loads(pickle.dumps(ef)) 			
		# 1. Import the "pickle" module.
		# 2. Serialize the "ef" object using "pickle.dumps".
		# 3. Deserialize the serialized object using "pickle.loads".
		# 4. Create a new object called "ef_max_sharpe" from the deserialized object.
		# This method is useful when you want to make a duplicate of an object without 
		# modifying the original object. It is also useful when you want to store an object in a file or send it over a network.
		# original method was to use copy.deepcopy(ef) but this breaks the code on cloud deployment. Cloud does not support deepcopy of CVXPY expression 
		# however we need to use deepcopy because the original object is modified when we call ef.max_sharpe() and we need to keep the original object intact
		# so we use pickle.loads(pickle.dumps(ef)) to make a copy of the object
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe(risk_free_rate) # risk_free_rate is the risk-free rate of return which is the return you would get if you invested in a risk-free asset like a US Treasury Bill
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe") # s is size of marker, c is color of marker
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig

# The code to get stock prices using yfinance is below and in a try/except block because it sometimes fails and we need to catch the error
# the try block will try to run the code in the try block. If it fails, it will run the code in the except block
# the except block will run if the code in the try block fails
try:
	# Get Stock Prices using pandas_datareader Library	
	stocks_df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
	stocks_df = stocks_df
	# # Plot Individual Stock Prices
	fig_price = px.line(stocks_df, title='Price of Individual Stocks')
	# # Plot Individual Cumulative Returns
	fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
	# # Calculatge and Plot Correlation Matrix between Stocks
	corr_df = stocks_df.corr().round(2) # round to 2 decimal places
	fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks')

	# Calculate expected returns and sample covariance matrix for portfolio optimization later
	mu = expected_returns.mean_historical_return(stocks_df)
	S = risk_models.sample_cov(stocks_df)

	# Plot efficient frontier curve
	fig = plot_efficient_frontier_and_max_sharpe(mu, S)
	fig_efficient_frontier = BytesIO()
	fig.savefig(fig_efficient_frontier, format="png")

	# Get optimized weights
	ef = EfficientFrontier(mu, S)
	ef.add_objective(objective_functions.L2_reg, gamma=0.1)
	# ef.add_constraint(lambda w: all(wi >= 0.05 for wi in w)) # delete
	# ef.add_constraint(lambda w: sum(w) == 1) # delete
	ef.max_sharpe(risk_free_rate)
	weights = ef.clean_weights()

	expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
	weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
	weights_df.columns = ['weights']

	# Calculate returns of portfolio with optimized weights by multiplying the 
	# weights by the returns of each stock and saving it in a new column of the stocks_df dataframe
	stocks_df['Optimized Portfolio'] = 0
	for ticker, weight in weights.items():
		stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight

	# Download the weights_df dataframe as a csv file using a button
	# @st.cache # this is a decorator that caches the function so that it doesn't have to be rerun every time the app is run
	# def convert_df(weights_df): # this function converts the weights_df dataframe to a csv file
	# # code to create or retrieve the weights_df dataframe goes here
	# 	return weights_df.to_csv().encode('utf-8')
	# csv = convert_df(weights_df) # assign the output of the convert_df function to a variable called csv

	# st.download_button( # this creates a download button
	# label="Download Optimized Weights as CSV",
	# data=csv,
	# file_name='weights_df.csv',
	# mime='text/csv')
	# st.markdown("""---""")

except:
	st.write(logging.exception(''))
	###st.write('Enter correct stock tickers to be included in portfolio separated\********************************************
# by commas WITHOUT spaces, e.g. "MA,FB,V,AMZN,JPM,BA"and hit Enter.')

#___________________________________________2/2/23 _________________________________________________________

stocks_df['Optimized Portfolio Amounts'] = 0
stocks_df2 = stocks_df
stocks_df2['Time'] = stocks_df2.index
for ticker, weight in weights.items():
	stocks_df2['Optimized Portfolio Amounts'] += stocks_df2[ticker]*(weight/100)*amount

# This code is to display how much the initial investment would be worth today
if pd.isna(stocks_df2["Optimized Portfolio Amounts"].iloc[-1]): # if the last value is NaN
    previous_date_value = stocks_df2["Optimized Portfolio Amounts"].iloc[-2] # if the last value is NaN, use the second to last value
else:
    previous_date_value = stocks_df2["Optimized Portfolio Amounts"].iloc[-1] # if the last value is not NaN, use the last value

st.markdown(f''' ###  If you would have invested $ *{amount:,.2f}* in the Optimized Portfolio on *{start_date}*
### it would be worth $ *{previous_date_value:,.2f}* today. :eyes: ''')
st.markdown("""---""")

# Optimized Portfolio: Cumulative Returns
fig = px.line(stocks_df2, x='Time', y='Optimized Portfolio Amounts', title= 'Optimized Portfolio: Cumulative Returns')
fig.update_yaxes(title_text='$ Amount')
st.plotly_chart(fig)

st.header('Performance Expectations:')
st.subheader('- Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
st.subheader('- Annual volatility: {}%'.format((annual_volatility*100).round(2)))
st.subheader('- Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
st.markdown("""---""")

# Tables of weights and amounts
col1, col2 = st.columns(2)
with col1:
	# display the weights_df dataframe
	st.markdown('''#### WEIGHTS 
				(must add up to 1) ''')
	weights_df['weights'] = (weights_df['weights']).round(2)
	weights_df = weights_df.sort_values(by=['weights'], ascending=False)
	st.dataframe(weights_df)
	
with col2:
	st.markdown(f'''#### BUY THIS AMOUNT 
				(must add up to $ {amount}) ''')
	# display the weights_df dataframe multiplied by the amount of money invested
	amounts = weights_df*amount
	amounts_sorted=amounts.sort_values(by=['weights'], ascending=False)
	# rename the weights column to amounts
	amounts_sorted.columns = ['$ amounts']
	st.dataframe(amounts_sorted)
	    
# st.plotly_chart(fig_cum_returns_optimized)

# show stocks_df dataframe
# st.dataframe(stocks_df2)

# st.header("Optimized Max Sharpe Portfolio Weights")
# st.dataframe(weights_df)

# add link to Correlation Matrix header that takes you to investopedia article on correlation
st.header('Correlation Matrix') # https://www.investopedia.com/terms/c/correlationcoefficient.asp
st.markdown('''[Correlation Info](https://www.investopedia.com/terms/c/correlationcoefficient.asp)''')
st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
st.markdown("""---""")





#################################################################################################################################
