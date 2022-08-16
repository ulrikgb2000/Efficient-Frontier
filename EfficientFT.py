# https://www.youtube.com/watch?v=mJTrQfzr0R4
# https://www.youtube.com/watch?v=f2BCmQBCwDs
from statistics import mean
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def portfolioreturn (weights):
    return np.dot(df.mean(), weights)

# Efficient var
def portfolioSD(weights):
    return (np.dot(np.dot(df.cov(), weights), weights))**(1/2)*(np.sqrt(250))

# Creating random weights
def generateWeights(df):
    rand = np.random.random(len(df.columns))
    rand /= rand.sum()
    return rand

tickers = ['ENTRA.OL', 'TOM', 'TSLA', 'CAT']
potentialPortfolios = 10000

df = yf.download(tickers, start="2010-12-01", end="2022-06-01")
df = np.log(1+df['Adj Close'].pct_change())
#print(df)

weight = np.zeros((potentialPortfolios, len(tickers)))
expectedReturn = np.zeros((potentialPortfolios, len(tickers)))
expectedVolatility = np.zeros((potentialPortfolios, len(tickers)))
sharpeRatio = np.zeros((potentialPortfolios, len(tickers)))

meanLogReturn = df.mean(axis=0)
covar = df.cov()

for portfolio in range(potentialPortfolios):
    # generate random weight vector
    w = generateWeights(df)
    weight[portfolio,:] = w
    # expected log return
    expectedReturn[portfolio] = np.sum(meanLogReturn * w)
    # expected volatility
    expectedVolatility[portfolio] = np.sqrt( np.dot(w.T, np.dot(covar, w)))
    # sharpe ratio
    sharpeRatio[portfolio] = expectedReturn[portfolio] / expectedVolatility[portfolio]

bestSharpeIndex = sharpeRatio.argmax()
#print(weight[bestSharpeIndex])
print(len(sharpeRatio))
plt.figure(figsize=(12,6))
plt.scatter(expectedVolatility, expectedReturn)
plt.xlabel("Expected volatility")
plt.ylabel("Expected return")
plt.show()

"""
pfReturns = []
stds = []
w = []

for i in range(500):
    weights = generateWeights(df)
    pfReturns.append(portfolioreturn(weights))
    stds.append(portfolioSD(weights))
    w.append(weights)

plt.scatter(stds, pfReturns)
# plotting one stocks
# plt.scatter(df.std().iloc[0]*np.sqrt(250), df.mean().iloc[0], c='k') 
# Lowest risk portfolio 
plt.scatter(min(stds), pfReturns[stds.index(min(stds))], c='g')
plt.title("Efficient frontier")
plt.xlabel("Portfoliostd")
plt.ylabel("Portfolioreturn")
plt.show()


min(stds)
"""