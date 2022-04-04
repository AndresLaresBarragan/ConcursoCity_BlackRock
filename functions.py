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


import yfinance as yf, datetime as dt, pandas as pd, plotly.express as px
import pandas_datareader.data as web, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm 
from scipy import optimize as opt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
 

class Activos:
    
    def __init__(self, tickers:'Claves de cotización [lista]'):
        """
        Inicialización de objeto.
        """
        self.tickers = tickers
    
    
        
    def get_adj_closes(self, 
                       start_date:"YYYY-MM-DD" = None, 
                       end_date:"YYYY-MM-DD" = None):
        """
        Método que obtiene los precios de cierre ajustados entre un rango de fechas establecido.
        inclusión de datos: [start_date, end_date]
        """
        # Descargamos DataFrame con todos los datos
        closes = web.DataReader(name = self.tickers, data_source='yahoo', start=start_date, end=end_date, )
        # Solo necesitamos los precios ajustados en el cierre
        closes = closes['Adj Close']
        # Se ordenan los índices de manera ascendente
        closes.sort_index(inplace=True)
        self.hist_prices = closes  # almacena datos hist.
        return closes
    
    
    
    def get_returns(self):
        "Obtiene rendimientos con la misma periodicidad que los datos"
        self.returns = self.hist_prices.pct_change().dropna()
        return self.returns
    
    
    
    def get_summary(self):
        """
        Obtiene el rendimineto esperado diario y volatilidad diaria de cada activo
        Su cálculo asume probabilidad uniforme en los rendimientos.
        """
        rends = self.returns.mean()
        vols  = self.returns.std()
        self.summary = pd.DataFrame({
            'Er':np.round(rends*252, 4),
            'Vol':np.round(vols*np.sqrt(252), 4)
        })
        return self.summary
    
    
    
    def plot_risk_return(self):
        """
        Grafica a cada activo en un punto dentro del espacio rendimiento-volatilidad.
        Ajusta la mejor recta de acuerdo al criterio de Min. Cuadrados
        """
        
        # Función de error de ajuste (mínimos cuadrados)
        def MSE(beta, x, y):
            recta = beta[0] + beta[1]*x
            errores = y - recta 
            return (errores**2).mean()
        
        # Aporximaciones iniciales
        beta_guess = [0,0]
        
        # Er y vol anualizados
        annual_summary = pd.DataFrame({
            "E[r]" : self.summary['Er'],
            "Vol" : self.summary['Vol']})
        
        # minimizar MSE
        sol = opt.minimize(
            fun = MSE,
            x0 = beta_guess,
            args = (annual_summary['Vol'],
                  annual_summary['E[r]']))
        
        beta = sol.x
        
        # Graficar en espacio rend-vol
        plt.figure(figsize=(14,10))
        plt.plot(annual_summary['Vol'],  # eje X
                 annual_summary['E[r]'], # eje Y
                 'ob', # marker type/color
                 ms = 8 # marker size
                )
        [plt.text(annual_summary.loc[stock,'Vol']+.001, annual_summary.loc[stock,'E[r]'], stock) for stock in annual_summary.index]
        
        x = np.linspace(annual_summary['Vol'].min(), annual_summary['Vol'].max())
        plt.plot(x, beta[0]+beta[1]*x, 'r--',lw = 2) # recta ajustada
        plt.title("Fondos de Inversión en espacio Media-Volatilidad")
        plt.xlabel("Volatilidad $\sigma$ (%)")
        plt.ylabel("Rendimiento esperado $E[r]$ (%)")
        plt.grid()
        plt.show()
        
        self.LinReg = sol

########## Func ejecutbale  ############

def histAnalysis(tickers, start_date="2018-03-01", end_date="2022-03-01"):
    
    portafolios = Activos(tickers)
    portafolios.get_adj_closes(start_date, end_date);
    portafolios.get_returns();
    portafolios.get_summary();
    
    #### resumen media y volatilidad ####
    display(portafolios.summary)
    
    ### Precios de Cierre Ajustados ###
    
    fig = make_subplots(rows=5, cols=1)
    fig.append_trace(go.Scatter(
        x=portafolios.hist_prices[tickers[0]].dropna().index,
        y=portafolios.hist_prices[tickers[0]].dropna(),
        name=f'{tickers[0]}'
    ), row=1, col=1)
    
    fig.append_trace(go.Scatter(
        x=portafolios.hist_prices[tickers[1]].dropna().index,
        y=portafolios.hist_prices[tickers[1]].dropna(),
        name=f'{tickers[1]}'
    ), row=2, col=1)
    
    fig.append_trace(go.Scatter(
        x=portafolios.hist_prices[tickers[2]].dropna().index,
        y=portafolios.hist_prices[tickers[2]].dropna(),
        name=f'{tickers[2]}'
    ), row=3, col=1)
    
    fig.append_trace(go.Scatter(
        x=portafolios.hist_prices[tickers[3]].dropna().index,
        y=portafolios.hist_prices[tickers[3]].dropna(),
        name=f'{tickers[3]}'
    ), row=4, col=1)
    
    fig.append_trace(go.Scatter(
        x=portafolios.hist_prices[tickers[4]].dropna().index,
        y=portafolios.hist_prices[tickers[4]].dropna(),
        name=f'{tickers[4]}', 
    ), row=5, col=1)
    
    fig.update_layout(height=600, width=900, title_text="Precios de Cierre Ajustados")
    fig.show()
    
    
    #### Rendiemintos diarios ####
    
    fig = make_subplots(rows=5, cols=1)
    fig.append_trace(go.Scatter(
        x=portafolios.returns[tickers[0]].dropna().index,
        y=portafolios.returns[tickers[0]].dropna(),
        name=f'{tickers[0]}'
    ), row=1, col=1)
    
    fig.append_trace(go.Scatter(
        x=portafolios.returns[tickers[1]].dropna().index,
        y=portafolios.returns[tickers[1]].dropna(),
        name=f'{tickers[1]}'
    ), row=2, col=1)
    
    fig.append_trace(go.Scatter(
        x=portafolios.returns[tickers[2]].dropna().index,
        y=portafolios.returns[tickers[2]].dropna(),
        name=f'{tickers[2]}'
    ), row=3, col=1)
    
    fig.append_trace(go.Scatter(
        x=portafolios.returns[tickers[3]].dropna().index,
        y=portafolios.returns[tickers[3]].dropna(),
        name=f'{tickers[3]}'
    ), row=4, col=1)
    
    fig.append_trace(go.Scatter(
        x=portafolios.returns[tickers[4]].dropna().index,
        y=portafolios.returns[tickers[4]].dropna(),
        name=f'{tickers[4]}', 
    ), row=5, col=1)
    
    fig.update_layout(height=750, width=900, title_text="Rendimientos dairios")
    fig.show()
    
    
    #### Análisis de media-varianza ####
    Activos.plot_risk_return(portafolios)
    