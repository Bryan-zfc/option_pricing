import numpy as np
import math
import yfinance as yf
from scipy.stats import norm
from datetime import date




#S_0 is the current stock price, K is the strike price, T is the expiration date, r is the annualised risk free rate and sigma is 
# the standard deviation of the stock's returns.
def black_scholes_option_price(S_0, K, T, r, sigma, option_type = "call"):

    d_plus = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    d_minus = d_plus - sigma  * np.sqrt(T)

    if option_type.lower() == "call":
        price = S＿0 * norm.cdf(d＿plus) - K * np.exp(-r*T) * norm.cdf(d_minus)
    elif option_type.lower() == "put":
        price = K * np.exp(-r*T) * norm.cdf(-d_minus) - S_0 * norm.cdf(-d_plus)
    return price 

def mc_black_scholes_option_price(S_0, K, T, r, sigma, n_sims=1000, option_type="call"):
    """
    Monte Carlo option pricing using GBM.
    """
    Z = np.random.normal(0, 1, n_sims)
    ST = S_0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0)
    elif option_type.lower() == "put":
        payoff = np.maximum(K - ST, 0)

    price = np.exp(-r*T) * payoff.mean()
    stderr = payoff.std(ddof=1) / np.sqrt(n_sims)

    return price, stderr


# 1. Download historical data
ticker = "AAL"
data = yf.download(ticker, start="2020-01-01", end=None,  auto_adjust=False)['Adj Close']

# 2. Compute daily log returns
returns = np.log(data / data.shift(1)).dropna()

# 3. Estimate historical volatility (annualized)
sigma = returns.std().item() * math.sqrt(252)#252 trading days
print("Annualized volatility:", sigma)

# 4. Current price
S0 = data.iloc[-1].item()
K = 13      # strike
expiration_date = date(2026,5,14)
today = date.today()
option_type = "put"
T = (expiration_date - today).days / 365     
r = 0.047      # risk-free rate
number_of_simulations = 10000

bs_call = black_scholes_option_price(S0, K, T, r, sigma, option_type)
mc_call, mc_err = mc_black_scholes_option_price(S0, K, T, r, sigma, number_of_simulations ,option_type)

print("Analytic Call Price:", bs_call)
print("Monte Carlo Call Price:", mc_call, "+/-", mc_err)
