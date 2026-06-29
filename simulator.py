import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

aapl_data=yf.download("AAPL",start="2022-01-01",end="2022-12-31")
print(aapl_data.head())

closeprices=aapl_data["Close"]
print(closeprices)

dailyreturns=closeprices.pct_change().dropna();
mu=dailyreturns.mean().values[0];
sigma=dailyreturns.std().values[0];
print(f"Daily mean return: {mu}");
print(f"Daily volatility: {sigma}");

paths=1000;
S0=float(closeprices.iloc[-1].values[0]);
days=252;
dt=1/252;
Z=np.random.normal(0,1,size=(days,paths));
pricepaths=np.zeros((days,paths));
pricepaths[0]=S0;

for t in range(1,days):
    drift = (mu - 0.5 * sigma**2)
    shock = sigma * Z[t]
    pricepaths[t] = pricepaths[t-1] * np.exp(drift + shock)

plt.figure(figsize=(10, 6))
plt.plot(pricepaths)
plt.title(f"Monte Carlo Simulation for AAPL ({paths} paths)")
plt.xlabel("Days")
plt.ylabel("Simulated Price")
plt.show()

finalprice=pricepaths[-1];
meanfinalprice=np.mean(finalprice);
ValueatRisk=np.percentile(finalprice,5);
profitprobability=np.mean(finalprice>S0);

print("Mean Final Price:", meanfinalprice);
print("Value at Risk: ", ValueatRisk);
print("Probability of profit: ", profitprobability);
