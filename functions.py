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
            
    