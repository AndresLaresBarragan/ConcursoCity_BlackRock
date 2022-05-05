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

from bs4 import BeautifulSoup
import urllib.request
import re
import json
import seaborn as sns

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm 
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, interactive
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Descarga de datos
def downloadData(tickers : "Símbolos de cotización"):
    data = pd.DataFrame()    
    
    for ticker in tickers:
        data[ticker] = yf.download(ticker, progress = False)["Adj Close"]
       
    return data

# Análisis descriptivo
def fundData(tickers: 'Nombre del ticker en lista', 
              links: 'Link del ticker en BlackRock en lista',
             colors: 'Lista de la paleta de colores a utilizar (seaborn)'):
    
    'Función que grafica la composición del fondo, su exposición geográfica y su exposición por sector.'
    
    def findOccurrences(s, ch):
        return [i for i, letter in enumerate(s) if letter == ch]
    
    for i in range(len(links)):
        
        # Descarga del html de la página
        source = urllib.request.urlopen(links[i]).read()
        soup = BeautifulSoup(source,'html.parser')
        
        # Tabla de composición del fondo
        table = soup.find_all('table')

        df1 = pd.read_html(str(table))[4]
        df2 = pd.read_html(str(table))[5]

        comp = pd.concat([df1, df2])[::-1]
        
        labels = comp["Nombre"]
        data = comp["Peso (%)"] 

        # Visualización
        plt.style.use('seaborn')
        fig = plt.figure(figsize = (16, 14), constrained_layout = True)
        
        # Gráfica de la composición del fondo
        spec = fig.add_gridspec(3, 2)
        fig.suptitle(tickers[i], fontweight='bold')
        
        colors_comp = sns.color_palette(colors[i], len(comp))
        
        ax0 = fig.add_subplot(spec[0, :])
        ax0.barh(labels, data, align = 'center', color = colors_comp) 
        
        for j, v in enumerate(sorted(data)):
            plt.text(v + 0.2, j, str(round(v, 2)), color = 'black', va = "center", weight = 'bold')
        
        # Información del sector y región de exposición del fondo
        sector = soup.find(string=re.compile('var tabsSectorDataTable')) 
        region = soup.find(string=re.compile('var subTabsCountriesDataTable')) 

        data_sector = sector[findOccurrences(sector, '[')[2]:findOccurrences(sector, ']')[0] + 1]
        data_region = region[findOccurrences(region, '[')[2]:findOccurrences(region, ']')[0] + 1]
        
        indexes_sector = [x.start() for x in re.finditer('\,}', data_sector)]
        indexes_region = [x.start() for x in re.finditer('\,}', data_region)]
        
        data_sector = "".join([char for idx, char in enumerate(data_sector) if idx not in indexes_sector])
        data_region = "".join([char for idx, char in enumerate(data_region) if idx not in indexes_region])
        
        df_sector = pd.DataFrame(json.loads(data_sector))
        df_region = pd.DataFrame(json.loads(data_region))
        
        df_sector = df_sector[df_sector["value"].str.contains("0.00")==False]
        df_region = df_region[df_region["value"].str.contains("0.00")==False]
        
        # Gráficas de exposición por sector y por región
        colors_sector = sns.color_palette(colors[i], len(df_sector))[::-1]
        colors_region = sns.color_palette(colors[i], len(df_region))[::-1]
        
        ax10 = fig.add_subplot(spec[1, 0])
        ax10.pie(df_sector.iloc[:,1], autopct='%.1f%%', pctdistance=1.1, colors=colors_sector, #labeldistance=1.4,
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
               textprops={'size': 'large'});
        ax10.axis('equal')
        ax10.legend(df_sector.iloc[:,0], loc='center left', bbox_to_anchor=(1, 0.5), shadow = True)
        plt.title('Exposición a sectores', fontweight='bold')
        
        ax20 = fig.add_subplot(spec[2, 0])
        ax20.pie(df_region.iloc[:,1], autopct='%.1f%%', pctdistance=1.1, colors=colors_region, #labeldistance=1.4,
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
               textprops={'size': 'large'});
        ax20.axis('equal')
        ax20.legend(df_region.iloc[:,0], loc='center left', bbox_to_anchor=(1, 0.5), shadow = True)
        plt.title("Exposición geográfica", fontweight='bold')
        

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
    
    
######### -------------------------------------   ETF's vs Factores ----------------------------------------

def etf_equity(variables):

    data = pd.DataFrame()

    for variable in variables:
        data[variable] = yf.download(variable, progress = False, interval = "1mo")["Adj Close"]

    dataReturns = data.pct_change().dropna()
    
    X = dataReturns.iloc[:, 1:] 
    Y = dataReturns.iloc[:, 0]
    X = sm.add_constant(X) 
    
    model = sm.OLS(Y, X).fit()

    return model, dataReturns

def model_coef(model):

    data = model.params
    values = data.tolist()
    variables = data.index.tolist()
    y_pos = np.arange(len(variables))
    
    results_as_html = model.summary().tables[0].as_html()
    table = pd.read_html(results_as_html, header=0, index_col=0)[0]
    title = table.columns.tolist()[0]

    fig = go.Figure(go.Bar(
                x=values,
                y=variables,
                orientation='h'))

    fig.update_layout(title = "Sensibilidad a Factores: " + title)
        
    return fig

def etfButton():
    # ETF's seleccionados
    etfs = ["BLKDINB1-A.MX", "BLKCORB0-D.MX", "GOLD5+B2-C.MX", "BLKINT1B1-D.MX", "BLKUSEQB1-C.MX"]
    ETFButton = widgets.Dropdown(options = etfs, description = "ETF's")
    
    return ETFButton

def etfFactorsAnalysis(ETFButton):
    
    # ETF's de renta variable
    if ETFButton.value == "GOLD5+B2-C.MX" or ETFButton.value == "BLKINT1B1-D.MX" or ETFButton.value == "BLKUSEQB1-C.MX":
        variables2 = ["IEF", "LQD", "UUP", "GSG", "HYG", "^VIX"]
        if ETFButton.value == "GOLD5+B2-C.MX":
            variables = [ETFButton.value] + ["^MXX", "^GSPC", "ACWI", "MXN=X"] + variables2
            
        elif ETFButton.value == "BLKINT1B1-D.MX":
            variables = [ETFButton.value] + ["^GSPC", "VEA", "MXN=X"] + variables2
            
        else:
            variables = [ETFButton.value] + ["^GSPC", "MXN=X"] + variables2
            
        # Regresión lineal    
        model, data = etf_equity(variables)
        yhat = model.predict()
        r2 = str(round(model.rsquared_adj, 2) * 100)
        
        # Visuales
        fig1 = make_subplots(specs=[[{"secondary_y": False}]])
        fig1.add_trace(go.Scatter(x = data.index, y = data.iloc[:, 0], name = "Rendimientos"), 
                       secondary_y = False,)
        fig1.add_trace(go.Scatter(x = data.index, y = yhat, 
                                  name = "Predicción " + ", R-squared = " + r2 + "%"), 
                       secondary_y = False,)
        fig1.update_layout(title = "Regresión Lineal: " +  ETFButton.value + " ETF vs Factores",  xaxis_title = "Fecha")
        fig1.update_yaxes(title_text = "Retornos", secondary_y = False)        
        fig1.show()
        
        fig2 = model_coef(model)
        fig2.show()
        
        return model, data
        
    # ETF's de renta fija
    else:
        return 0
    
    
def factorsButton(data):
    # Factores 
    factors = list(data.iloc[:, 1:].columns)
    FactorsButton = widgets.Dropdown(options = factors, description = "Factores")
    
    return FactorsButton

def factorsVisual(data, ETFButton, FactorsButton):
    # Regresión lineal con el factor seleccionado
    
    X = data[FactorsButton.value]
    Y = data[ETFButton.value]
    X = sm.add_constant(X) 
    
    model = sm.OLS(Y, X).fit()
    yhat = np.dot(X.values, model.params.values)
    
    fig1 = make_subplots(specs=[[{"secondary_y": False}]])
    fig1.add_trace(go.Scatter(x = X.iloc[:, 1], y = yhat, name = "Recta Ajustada"), 
                       secondary_y = False,)
    fig1.add_trace(go.Scatter(x = X.iloc[:, 1], y = Y, 
                                  name = "ETF vs Factor", mode = "markers"), secondary_y = False,)
    
    fig1.update_layout(title = "Regresión Lineal: " +  ETFButton.value + " ETF vs Factor " + FactorsButton.value,  
                       xaxis_title = "Retornos Factor")
    fig1.update_yaxes(title_text = "Retornos ETF", secondary_y = False)  
    fig1.show()

    