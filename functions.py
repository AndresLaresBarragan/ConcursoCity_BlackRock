"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Proyecto de Investigación Citibanamex: BlackRock.                                          -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author:                                                                                             -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/diegolazareno/ConcursoCity_BlackRock                                 -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# Librerías requeridas
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats as st

# Descarga de datos
def downloadData(tickers : "Símbolos de cotización"):
    data = pd.DataFrame()    
    
    for ticker in tickers:
        data[ticker] = yf.download(ticker, progress = False)["Adj Close"]
       
    return data

# Análisis descriptivo

# Análisis histórico

# Análisis del riesgo
def riskAnalysis(historicPrices : "DataFrame con los precios históricos de los ETF's",
                confidenceLevel : "Nivel de confianza", colors : "colores", significanceLevel : "Nivel de significancia"):
    
    for i in range(len(historicPrices.iloc[0])):
        plt.style.use('seaborn')
    
        # Rendimientos
        returns = historicPrices.iloc[:, i].dropna().pct_change().dropna()
    
        # VaR y Expected Shortfall
        VaR = np.percentile(returns, 100 - confidenceLevel) 
        ES = np.mean(returns[returns < VaR])
        
        # Simulación de precios
        mu, sigma = np.mean(returns), np.std(returns)
        idx = pd.bdate_range(returns.index[-1] + dt.timedelta(1), end = returns.index[-1] + dt.timedelta(365))
        d_t = np.arange(1, len(idx) + 1)
        expectedPrice = pd.Series(historicPrices.iloc[-1, i] * np.exp((mu - (sigma ** 2) / 2) * d_t), idx)
        
        # Intervalos de confianza
        Z = st.norm.ppf(1 - significanceLevel / 2)
        infLim = pd.Series(np.exp(np.log(historicPrices.iloc[-1, i]) + (mu - (sigma ** 2) / 2) * d_t - Z * (sigma * np.sqrt(d_t))), idx)
        supLim = pd.Series(np.exp(np.log(historicPrices.iloc[-1, i]) + (mu - (sigma ** 2) / 2) * d_t + Z * (sigma * np.sqrt(d_t))), idx)
    
        # Visualización
        fig = plt.figure(figsize = (14, 12), constrained_layout = True)
        spec = fig.add_gridspec(3, 2)
        fig.suptitle("Simulación & Value at Risk | Expected Shortfall: " + historicPrices.columns[i])
        
        ax0 = fig.add_subplot(spec[0, :])
        ax0.plot(historicPrices.iloc[:, i], color = colors[i], label = "Cierre")
        ax0.plot(expectedPrice, "--", color = "k", label = "E[Precio]: $" + str(round(expectedPrice[-1], 2)))
        ax0.fill_between(expectedPrice.index, infLim, supLim, color = colors[i], alpha = 0.25, 
                          label = "Intervalo confianza " + 
                         str(100 - significanceLevel * 100) + "% " + "(" + str(round(infLim[-1], 2)) + "-" 
                         + str(round(supLim[-1], 2)) + ")")
        ax0.legend(loc = "upper left")
        ax0.set_xlabel("Fecha")
        ax0.set_ylabel("$ (MXN)")
        
        ax10 = fig.add_subplot(spec[1, 0])
        ax10.hist(returns, bins = 30, density = True, alpha = 0.35, color = colors[i])
        ax10.axvline(x = VaR, label = "VaR " + str(confidenceLevel) + "% : " + str(round(VaR * 100, 2)) + "%", linestyle = "--", color = colors[i])
        ax10.axvline(x = ES, label = "ES " + str(confidenceLevel) + "% : " + str(round(ES * 100, 2)) + "%", linestyle = "--", color = "k")
        ax10.set_xlabel("Rendimientos")
        ax10.legend()
    
        # Backtesting
        returns = pd.DataFrame(returns)
        returns["VaR%"] = np.nan
        returns["Expected Shortfall"] = np.nan

        j = 0
        for k in range(len(returns)):    
            if returns.index[k] >= pd.to_datetime("2020-01-01"): 
                returns.iloc[k, 1] = np.percentile(returns.iloc[j : k, 0], 100 - confidenceLevel)
                returns.iloc[k, 2] = np.mean(returns.iloc[j : k, 0][returns.iloc[j : k, 0] < returns.iloc[k, 1]])
                j += 1
        
        # Visualización
        returns.dropna(inplace = True)
        ax11 = fig.add_subplot(spec[1, 1])
        ax11.plot(returns.iloc[:, 0], label = "Retornos", color = colors[i], alpha = 0.35)
        ax11.plot(returns.iloc[:, 1], label = "Backtesting: VaR", color = colors[i])
        ax11.plot(returns.iloc[:, 2], label = "Backtesting: ES", color = "k")
        ax11.set_ylabel("%")
        ax11.set_xlabel("Fecha")
        ax11.legend()
        
        # Drawdown
        V = historicPrices.iloc[:, i]
        ax2 = fig.add_subplot(spec[2, :])
        ax2.fill_between(V.index, (V - V.cummax()) / V.cummax(), color = colors[i], alpha = 0.25, 
                          label = "Drawdown")
        ax2.legend(loc = "lower left")
        ax2.set_xlabel("Fecha")
        ax2.set_ylabel("%")