# https://www.youtube.com/watch?v=mJTrQfzr0R4
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

df = yf.download(['TSLA', 'CAT'], start="2010-12-01", end="2022-06-01")

df = np.log(1+df['Adj Close'].pct_change())

def portfolioreturn (weights):
    return np.dot(df.mean(), weights)

# Portfolio covariance and standard deviance
"""
df.cov()
portfolioVar = weights[0]**2*df.cov().iloc[0,0] + weights[1]**2*df.cov().iloc[1,1] + 2*weights[0]*weights[1]*df.cov().iloc[0, 1]
portfolioSD = portfolioVar ** (1/2)
print (portfolioVar, portfolioSD)
"""

# Efficient var
def portfolioSD(weights):
    return (np.dot(np.dot(df.cov(), weights), weights))**(1/2)*(np.sqrt(250))

# Creating random weights
def generateWeights(df):
    rand = np.random.random(len(df.columns))
    rand /= rand.sum()
    return rand

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