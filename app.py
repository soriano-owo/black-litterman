import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

# --- Configuración de los ETFs disponibles ---
tickers_info = {
    "TLT": {
        "nombre": "iShares 20+ Year Treasury Bond ETF",
        "descripcion": "Sigue el índice ICE U.S. Treasury 20+ Year Bond Index, compuesto por bonos del gobierno de EE. UU. con vencimientos superiores a 20 años.",
        "indice": "ICE U.S. Treasury 20+ Year Bond Index",
        "sector": "Renta fija",
        "categoria": "Bonos del Tesoro de EE. UU.",
        "exposicion": "Bonos del gobierno de EE. UU. a largo plazo.",
        "exposicion_ganada": "Exposición a bonos de largo plazo altamente líquidos respaldados por el gobierno estadounidense.",
        "pais_inversion": ["Estados Unidos"],
        "moneda": "USD",
        "beta": 0.2,
        "duracion": "Larga",
        "top_holdings": [{"symbol": "US Treasury", "holdingPercent": "100%"}],
        "gastos": "0.15%",
        "rango_1y": "120-155 USD",
        "rendimiento_ytd": "5%",
        "estilo": "Grado de inversión",
    },
    "EMB": {
        "nombre": "iShares JP Morgan USD Emerging Markets Bond ETF",
        "descripcion": "Rastrea bonos soberanos de mercados emergentes denominados en dólares estadounidenses.",
        "indice": "J.P. Morgan EMBI Global Diversified Index",
        "sector": "Renta fija",
        "categoria": "Bonos emergentes",
        "exposicion": "Bonos soberanos de mercados emergentes denominados en USD.",
        "exposicion_ganada": "Acceso a bonos soberanos diversificados de mercados emergentes.",
        "pais_inversion": ["Brasil", "México", "Otros mercados emergentes"],
        "moneda": "USD",
        "beta": 0.6,
        "duracion": "Media",
        "top_holdings": [
            {"symbol": "Brazil 10Yr Bond", "holdingPercent": "10%"},
            {"symbol": "Mexico 10Yr Bond", "holdingPercent": "9%"}
        ],
        "gastos": "0.39%",
        "rango_1y": "85-105 USD",
        "rendimiento_ytd": "8%",
        "estilo": "Riesgo moderado",
    },
    "SPY": {
        "nombre": "SPDR S&P 500 ETF Trust",
        "descripcion": "Este ETF sigue el índice S&P 500, compuesto por las 500 principales empresas de EE. UU.",
        "indice": "S&P 500 Index",
        "sector": "Renta variable",
        "categoria": "Acciones grandes de EE. UU.",
        "exposicion": "Acciones de las 500 empresas más grandes de EE. UU.",
        "moneda": "USD",
        "beta": 1.0,
        "top_holdings": [
            {"symbol": "Apple", "holdingPercent": "6.5%"},
            {"symbol": "Microsoft", "holdingPercent": "5.7%"}
        ],
        "gastos": "0.0945%",
        "rango_1y": "360-420 USD",
        "rendimiento_ytd": "15%",
        "duracion": "Baja",
        "pais_inversion": ["Estados Unidos"],
        "estilo": "Blend",
    },
    "VWO": {
        "nombre": "Vanguard FTSE Emerging Markets ETF",
        "descripcion": "Sigue el índice FTSE Emerging Markets All Cap China A Inclusion Index.",
        "sector": "Renta variable",
        "categoria": "Acciones emergentes",
        "exposicion": "Mercados emergentes globales.",
        "exposicion_ganada": "Acceso diversificado a mercados emergentes globales.",
        "pais_inversion": ["China", "India", "Brasil", "Taiwán"],
        "moneda": "USD",
        "beta": 1.2,
        "top_holdings": [
            {"symbol": "Tencent", "holdingPercent": "6%"},
            {"symbol": "Alibaba", "holdingPercent": "4.5%"}
        ],
        "gastos": "0.08%",
        "rango_1y": "40-55 USD",
        "rendimiento_ytd": "10%",
        "duracion": "Alta",
        "estilo": "Crecimiento",
    },
    "GLD": {
        "nombre": "SPDR Gold Shares",
        "descripcion": "Este ETF sigue el precio del oro físico.",
        "sector": "Materias primas",
        "categoria": "Oro físico",
        "exposicion": "Oro físico.",
        "exposicion_ganada": "Exposición directa al precio del oro como activo refugio.",
        "pais_inversion": ["Global"],
        "moneda": "USD",
        "beta": 0.1,
        "top_holdings": [{"symbol": "Gold", "holdingPercent": "100%"}],
        "gastos": "0.40%",
        "rango_1y": "160-200 USD",
        "rendimiento_ytd": "12%",
        "duracion": "Baja",
        "estilo": "Activo refugio",
    }
}

# --- Funciones Auxiliares Científicas ---
def cargar_datos(tickers_list, inicio, fin):
    datos = {}
    for t in tickers_list:
        df = yf.download(t, start=inicio, end=fin, auto_adjust=True)
        if not df.empty:
            close_series = df["Close"].squeeze()
            df_clean = pd.DataFrame(index=df.index)
            df_clean["Close"] = close_series
            df_clean["Retornos"] = close_series.pct_change()
            datos[t] = df_clean
    return datos

def calcular_drawdown_y_watermark(precios):
    watermark = precios.cummax()
    drawdown = (precios / watermark) - 1
    return drawdown, watermark

def optimizar_portafolio_markowitz(retornos, metodo="min_vol"):
    media_anual = retornos.mean() * 252
    cov_anual = retornos.cov() * 252
    n = len(media_anual)
    w_inicial = np.ones(n) / n
    
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    limites = [(0, 1) for _ in range(n)]

    def riesgo(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_anual, w)))

    def sharpe_negativo(w):
        vol = riesgo(w)
        ret = np.dot(w.T, media_anual)
        return -(ret / vol) if vol > 0 else 0

    objetivo = sharpe_negativo if metodo == "sharpe" else riesgo
    resultado = minimize(objetivo, w_inicial, constraints=restricciones, bounds=limites, method="SLSQP")
    return np.array(resultado.x).flatten()

def black_litterman_optimizar(retornos, P, Q, tau=0.025):
    media_anual = retornos.mean().values * 252
    cov_anual = retornos.cov().values * 252
    
    omega = np.diag(np.diag(np.dot(np.dot(P, tau * cov_anual), P.T)))
    if np.all(omega == 0): 
        omega = np.eye(P.shape[0]) * 0.001

    try:
        inv_tau_cov = np.linalg.inv(tau * cov_anual)
        inv_omega = np.linalg.inv(omega)
        M = np.linalg.inv(inv_tau_cov + np.dot(np.dot(P.T, inv_omega), P))
        ajustada_media = np.dot(M, np.dot(inv_tau_cov, media_anual) + np.dot(np.dot(P.T, inv_omega), Q))
    except np.linalg.LinAlgError:
        ajustada_media = media_anual

    n = len(ajustada_media)
    w_inicial = np.ones(n) / n
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    limites = [(0, 1) for _ in range(n)]

    def sharpe_bl(w):
        ret = np.dot(w, ajustada_media)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_anual, w)))
        return -(ret / vol) if vol != 0 else 0

    resultado = minimize(sharpe_bl, w_inicial, constraints=restricciones, bounds=limites, method="SLSQP")
    return np.array(resultado.x).flatten()

def calcular_frontera_eficiente(retornos, num_puntos=1000):
    media_anual = retornos.mean() * 252
    cov_anual = retornos.cov() * 252
    n_activos = len(media_anual)
    resultados = {"port_rets": [], "port_vols": [], "sharpe_ratio": []}

    for _ in range(num_puntos):
        pesos = np.random.random(n_activos)
        pesos /= np.sum(pesos)
        rendimiento = np.dot(pesos, media_anual)
        riesgo = np.sqrt(np.dot(pesos.T, np.dot(cov_anual, pesos)))
        sharpe = rendimiento / riesgo if riesgo != 0 else 0
        
        resultados["port_rets"].append(rendimiento)
        resultados["port_vols"].append(riesgo)
        resultados["sharpe_ratio"].append(sharpe)

    return pd.DataFrame(resultados)

# --- Controles de Usuario ---
st.sidebar.title("Parámetros Cuantitativos")
tickers_seleccionados = st.sidebar.multiselect(
    "Selecciona los ETFs de la Cartera",
    options=list(tickers_info.keys()),
    default=["TLT", "EMB", "SPY", "GLD"]
)

if len(tickers_seleccionados) < 2:
    st.sidebar.error("Por favor selecciona al menos 2 ETFs.")
    st.stop()

fecha_inicio = st.sidebar.date_input("Inicio del Historial", value=date(2012, 1, 1))
fecha_fin_entrenamiento = st.sidebar.date_input("Fin del Entrenamiento (In-Sample)", value=date(2021, 1, 1))
fecha_inicio_backtest = st.sidebar.date_input("Inicio del Backtest (Out-of-Sample)", value=date(2021, 1, 2))
fecha_fin_backtest = st.sidebar.date_input("Fin del Backtest", value=date(2025, 1, 1))

if fecha_fin_entrenamiento <= fecha_inicio or fecha_fin_backtest <= fecha_inicio_backtest:
    st.sidebar.error("Error en la jerarquía de fechas configuradas.")
    st.stop()

tickers = {k: tickers_info[k] for k in tickers_seleccionados}

# --- App Principal ---
st.title("Asset Allocation & Portfolio Optimization Engine")
tabs = st.tabs(["Dashboard", "Análisis de Activos", "Frontera Eficiente", "Backtesting Histórico", "Black-Litterman Engine"])

# --- TAB 1: DASHBOARD ---
with tabs[0]:
    st.markdown("""
    Este entorno despliega herramientas analíticas de asignación de activos basándose en la **Teoría Moderna de Portafolio (MPT)** y el **Modelo de Optimización de Black-Litterman**. Permite analizar datos históricos (*In-Sample*) y contrastarlos mediante *Backtesting* con datos fuera de muestra (*Out-of-Sample*).
    """)
    st.info(f"**Universo de Inversión Activo:** {', '.join(tickers_seleccionados)}")

# --- TAB 2: ANÁLISIS DE ACTIVOS ---
with tabs[1]:
    st.header("Análisis de Series de Tiempo Estocásticas")
    datos_completos = cargar_datos(list(tickers.keys()), fecha_inicio.strftime("%Y-%m-%d"), fecha_fin_backtest.strftime("%Y-%m-%d"))
    
    st.dataframe(pd.DataFrame(tickers).T[["nombre", "sector", "categoria", "gastos", "beta"]])

    st.subheader("Métricas Estadísticas Anualizadas (Periodo de Entrenamiento)")
    for ticker in tickers.keys():
        df_act = datos_completos[ticker].loc[fecha_inicio.strftime("%Y-%m-%d"):fecha_fin_entrenamiento.strftime("%Y-%m-%d")]
        ret = df_act["Retornos"].dropna()
        
        ret_anual = ret.mean() * 252 * 100
        vol_anual = ret.std() * np.sqrt(252) * 100
        sharpe_anual = (ret.mean() * 252) / (ret.std() * np.sqrt(252)) if ret.std() != 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{ticker} Retorno Anual", f"{ret_anual:.2f}%")
        col2.metric(f"{ticker} Volatilidad Anual", f"{vol_anual:.2f}%")
        col3.metric(f"{ticker} Sharpe Ratio", f"{sharpe_anual:.2f}")

# --- TAB 3: PORTAFOLIOS ÓPTIMOS ---
with tabs[2]:
    st.header("Frontera Eficiente de Markowitz")
    df_retornos_entren = pd.DataFrame({k: datos_completos[k]["Retornos"] for k in tickers.keys()}).loc[fecha_inicio.strftime("%Y-%m-%d"):fecha_fin_entrenamiento.strftime("%Y-%m-%d")].dropna()
    
    pesos_min_vol = optimizar_portafolio_markowitz(df_retornos_entren, metodo="min_vol")
    pesos_sharpe = optimizar_portafolio_markowitz(df_retornos_entren, metodo="sharpe")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mínima Volatilidad")
        st.write({t: f"{p:.2%}" for t, p in zip(tickers.keys(), pesos_min_vol)})
    with col2:
        st.subheader("Máximo Sharpe Ratio")
        st.write({t: f"{p:.2%}" for t, p in zip(tickers.keys(), pesos_sharpe)})
        
    frontera_df = calcular_frontera_eficiente(df_retornos_entren)
    fig = px.scatter(frontera_df, x="port_vols", y="port_rets", color="sharpe_ratio", title="Espacio Riesgo-Retorno Anualizado", labels={"port_vols": "Volatilidad Anual", "port_rets": "Retorno Esperado Anual"})
    st.plotly_chart(fig)

# --- TAB 4: BACKTESTING ---
with tabs[3]:
    st.header("Backtesting Estricto Out-of-Sample")
    
    df_retornos_backtest = pd.DataFrame({k: datos_completos[k]["Retornos"] for k in tickers.keys()}).loc[fecha_inicio_backtest.strftime("%Y-%m-%d"):fecha_fin_backtest.strftime("%Y-%m-%d")].dropna()
    
    sp500 = yf.download("^GSPC", start=fecha_inicio_backtest.strftime("%Y-%m-%d"), end=fecha_fin_backtest.strftime("%Y-%m-%d"), auto_adjust=True)
    sp500_ret = sp500["Close"].squeeze().pct_change().dropna()
    
    sp500_ret = sp500_ret.reindex(df_retornos_backtest.index).fillna(0)
    
    eq_w = np.ones(len(tickers_seleccionados)) / len(tickers_seleccionados)
    
    ret_min_vol = df_retornos_backtest.dot(pesos_min_vol)
    ret_max_sh = df_retornos_backtest.dot(pesos_sharpe)
    ret_eq = df_retornos_backtest.dot(eq_w)
    
    df_cum_perf = pd.DataFrame({
        "Mínima Volatilidad": (1 + ret_min_vol).cumprod() - 1,
        "Máximo Sharpe": (1 + ret_max_sh).cumprod() - 1,
        "Equitativo (1/N)": (1 + ret_eq).cumprod() - 1,
        "S&P 500 Benchmark": (1 + sp500_ret).cumprod() - 1
    }, index=df_retornos_backtest.index)
    
    fig_backtest = px.line(df_cum_perf, title="Evolución de Retornos Compuestos Acumulados (Out-of-Sample)", labels={"value": "Retorno Acumulado", "Date": "Fecha"})
    st.plotly_chart(fig_backtest)

# --- TAB 5: BLACK-LITTERMAN ---
with tabs[4]:
    st.header("Black-Litterman Engine Core")
    st.markdown("Inserta tus expectativas de mercado anualizadas (*Views*) para reajustar los pesos de manera Bayesiana:")
    
    n_activos = len(tickers_seleccionados)
    P = np.eye(n_activos)
    
    Q_values = []
    for ticker in tickers_seleccionados:
        val = st.slider(f"View de Rendimiento Anual Esperado para {ticker}:", min_value=-15, max_value=30, value=6, step=1)
        Q_values.append(val / 100)
        
    Q = np.array(Q_values)
    
    pesos_bl = black_litterman_optimizar(df_retornos_entren, P, Q)
    
    st.subheader("Pesos Finales Ajustados por Black-Litterman")
    fig_bl = px.bar(x=list(tickers.keys()), y=pesos_bl, labels={"x": "ETF", "y": "Peso Optimizando"}, title="Distribución de Activos Final (BL)")
    st.plotly_chart(fig_bl)
    st.write({t: f"{p:.2%}" for t, p in zip(tickers.keys(), pesos_bl)})
