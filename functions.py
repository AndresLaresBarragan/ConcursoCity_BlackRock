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
def brownianMotion(historicPrices : "DataFrame con los precios históricos de los ETF's",
                significanceLevel : "Nivel de significancia %", colors : "colores"):
    plt.style.use('seaborn')

    for i in range(len(historicPrices.iloc[0])):
        
        # Parámetros históricos
        returns = historicPrices.iloc[:, i].dropna().pct_change().dropna()
        mu, sigma = np.mean(returns), np.std(returns)
    
        # Simulación de precios
        idx = pd.bdate_range(returns.index[-1] + dt.timedelta(1), end = returns.index[-1] + dt.timedelta(365))
        d_t = np.arange(1, len(idx) + 1)
        expectedPrice = pd.Series(historicPrices.iloc[-1, i] * np.exp((mu - (sigma ** 2) / 2) * d_t), idx)
        
        # Intervalos de confianza
        Z = st.norm.ppf(1 - significanceLevel / 2)
        infLim = pd.Series(np.exp(np.log(historicPrices.iloc[-1, i]) + (mu - (sigma ** 2) / 2) * d_t - Z * (sigma * np.sqrt(d_t))), idx)
        supLim = pd.Series(np.exp(np.log(historicPrices.iloc[-1, i]) + (mu - (sigma ** 2) / 2) * d_t + Z * (sigma * np.sqrt(d_t))), idx)
        
        # Visualización
        fig, axes = plt.subplots(1, 1, figsize = (15, 3.5))
        fig.suptitle("Simulación de precios: " + historicPrices.columns[i])
        
        axes.plot(historicPrices.iloc[:, i], color = colors[i], label = "Cierre")
        axes.plot(expectedPrice, "--", color = "k", label = "E[Precio]: $" + str(round(expectedPrice[-1], 2)))
        axes.fill_between(expectedPrice.index, infLim, supLim, color = colors[i], alpha = 0.25, 
                          label = "Intervalo confianza " + str(100 - significanceLevel * 100) + "%")
        axes.legend(loc = "upper left")
        axes.set_xlabel("Fecha")
        axes.set_ylabel("$ (MXN)")

def riskAnalysis(historicPrices : "DataFrame con los precios históricos de los ETF's",
                confidenceLevel : "Nivel de confianza", colors : "colores"):
    plt.style.use('seaborn')
    
    for i in range(len(historicPrices.iloc[0])):
    
        # Rendimientos
        returns = historicPrices.iloc[:, i].dropna().pct_change().dropna()
    
        # VaR y Expected Shortfall
        VaR = np.percentile(returns, 100 - confidenceLevel) 
        ES = np.mean(returns[returns < VaR])
    
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize = (15, 3.5))
        fig.suptitle("Value at Risk & Expected Shortfall: " + historicPrices.columns[i])
        
        axes[0].hist(returns, bins = 30, density = True, alpha = 0.35, color = colors[i])
        axes[0].axvline(x = VaR, label = "VaR " + str(confidenceLevel) + "% : " + str(round(VaR * 100, 2)) + "%", linestyle = "--", color = colors[i])
        axes[0].axvline(x = ES, label = "ES " + str(confidenceLevel) + "% : " + str(round(ES * 100, 2)) + "%", linestyle = "--", color = "k")
        axes[0].set_xlabel("Rendimientos")
        axes[0].legend()
    
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
        axes[1].plot(returns.iloc[:, 0], label = "Retornos", color = colors[i], alpha = 0.35)
        axes[1].plot(returns.iloc[:, 1], label = "Backtesting: VaR", color = colors[i])
        axes[1].plot(returns.iloc[:, 2], label = "Backtesting: ES", color = "k")
        axes[1].set_ylabel("%")
        axes[1].set_xlabel("Fecha")
        axes[1].legend()
            
    