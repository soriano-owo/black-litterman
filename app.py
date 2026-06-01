import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from scipy.optimize import minimize
import scipy.optimize as sco
from scipy.stats import skew, kurtosis

# --- Configuración de los ETFs disponibles ---
tickers_info = {
    "TLT": {
        "nombre": "iShares 20+ Year Treasury Bond ETF",
        "descripcion": "Este ETF sigue el índice ICE U.S. Treasury 20+ Year Bond Index, compuesto por bonos del gobierno de EE. UU. con vencimientos superiores a 20 años.",
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
        "descripcion": "Este ETF sigue el índice J.P. Morgan EMBI Global Diversified Index, que rastrea bonos soberanos de mercados emergentes en dólares estadounidenses.",
        "indice": "J.P. Morgan EMBI Global Diversified Index",
        "sector": "Renta fija",
        "categoria": "Bonos emergentes",
        "exposicion": "Bonos soberanos de mercados emergentes denominados en USD.",
        "exposicion_ganada": "Acceso a bonos soberanos diversificados de mercados emergentes.",
        "pais_inversion": ["Brasil", "México", "Rusia", "Otros mercados emergentes"],
        "moneda": "USD",
        "beta": 0.6,
        "duracion": "Media",
        "top_holdings": [
            {"symbol": "Brazil 10Yr Bond", "holdingPercent": "10%"},
            {"symbol": "Mexico 10Yr Bond", "holdingPercent": "9%"},
            {"symbol": "Russia 10Yr Bond", "holdingPercent": "7%"},
        ],
        "gastos": "0.39%",
        "rango_1y": "85-105 USD",
        "rendimiento_ytd": "8%",
        "estilo": "Riesgo moderado, rendimiento potencial",
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
            {"symbol": "Microsoft", "holdingPercent": "5.7%"},
            {"symbol": "Amazon", "holdingPercent": "4.3%"},
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
        "descripcion": "Este ETF sigue el índice FTSE Emerging Markets All Cap China A Inclusion Index.",
        "sector": "Renta variable",
        "categoria": "Acciones emergentes",
        "exposicion": "Mercados emergentes globales.",
        "exposicion_ganada": "Acceso diversificado a mercados emergentes globales.",
        "pais_inversion": ["China", "India", "Brasil", "Taiwán", "Otros emergentes"],
        "moneda": "USD",
        "beta": 1.2,
        "top_holdings": [
            {"symbol": "Tencent", "holdingPercent": "6%"},
            {"symbol": "Alibaba", "holdingPercent": "4.5%"},
            {"symbol": "Taiwan Semiconductor", "holdingPercent": "4%"},
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
        "exposicion": "Oro físico y contratos futuros de oro.",
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
    },
    "QQQ": {
        "nombre": "Invesco QQQ Trust",
        "descripcion": "Sigue el índice Nasdaq-100, compuesto por las 100 mayores empresas no financieras del Nasdaq.",
        "indice": "Nasdaq-100 Index",
        "sector": "Renta variable",
        "categoria": "Tecnología EE. UU.",
        "exposicion": "Las 100 mayores empresas no financieras del Nasdaq.",
        "exposicion_ganada": "Alta exposición al sector tecnológico de EE. UU.",
        "pais_inversion": ["Estados Unidos"],
        "moneda": "USD",
        "beta": 1.2,
        "top_holdings": [
            {"symbol": "Apple", "holdingPercent": "9%"},
            {"symbol": "Microsoft", "holdingPercent": "8%"},
            {"symbol": "Nvidia", "holdingPercent": "7%"},
        ],
        "gastos": "0.20%",
        "rango_1y": "350-500 USD",
        "rendimiento_ytd": "20%",
        "duracion": "Baja",
        "estilo": "Crecimiento",
    },
    "IEF": {
        "nombre": "iShares 7-10 Year Treasury Bond ETF",
        "descripcion": "Sigue bonos del Tesoro de EE. UU. con vencimientos de 7 a 10 años.",
        "indice": "ICE U.S. Treasury 7-10 Year Bond Index",
        "sector": "Renta fija",
        "categoria": "Bonos del Tesoro EE. UU.",
        "exposicion": "Bonos del gobierno de EE. UU. de mediano plazo.",
        "exposicion_ganada": "Exposición a bonos de mediano plazo con menor duración que TLT.",
        "pais_inversion": ["Estados Unidos"],
        "moneda": "USD",
        "beta": 0.1,
        "top_holdings": [{"symbol": "US Treasury", "holdingPercent": "100%"}],
        "gastos": "0.15%",
        "rango_1y": "90-105 USD",
        "rendimiento_ytd": "3%",
        "duracion": "Media",
        "estilo": "Grado de inversión",
    },
    "VNQ": {
        "nombre": "Vanguard Real Estate ETF",
        "descripcion": "Sigue el índice MSCI US Investable Market Real Estate 25/50, compuesto por REITs.",
        "indice": "MSCI US Investable Market Real Estate 25/50",
        "sector": "Bienes raíces",
        "categoria": "REITs EE. UU.",
        "exposicion": "Fideicomisos de inversión inmobiliaria (REITs) de EE. UU.",
        "exposicion_ganada": "Exposición al mercado inmobiliario sin comprar propiedades directamente.",
        "pais_inversion": ["Estados Unidos"],
        "moneda": "USD",
        "beta": 0.8,
        "top_holdings": [
            {"symbol": "Vanguard Real Estate II", "holdingPercent": "12%"},
            {"symbol": "American Tower", "holdingPercent": "6%"},
        ],
        "gastos": "0.12%",
        "rango_1y": "75-95 USD",
        "rendimiento_ytd": "7%",
        "duracion": "Media",
        "estilo": "Ingreso",
    },
}

# --- Funciones Auxiliares ---
def cargar_datos(tickers_list, inicio, fin):
    datos = {}
    for ticker in tickers_list:
        df = yf.download(ticker, start=inicio, end=fin, auto_adjust=True)
        df["Retornos"] = df["Close"].pct_change()
        datos[ticker] = df
    return datos


def graficar_linea(x_column, y_column, title, labels=None):
    if isinstance(y_column, pd.Series):
        y_column = y_column.values.flatten()
    fig = px.line(x=x_column, y=y_column, title=title, labels=labels)
    return fig


def calcular_drawdown_y_watermark(precios):
    watermark = precios.cummax()
    drawdown = (precios / watermark) - 1
    return drawdown, watermark


risk_aversion_lambda = 1
target_return = 0.10


def optimizar_portafolio_markowitz(retornos, metodo="min_vol", objetivo=None):
    media = retornos.mean()
    cov = retornos.cov()

    def riesgo(w):
        return np.sqrt(np.dot(w.T, np.dot(cov, w)))

    def sharpe(w):
        return -(np.dot(w.T, media) / np.sqrt(np.dot(w.T, np.dot(cov, w))))

    n = len(media)
    w_inicial = np.ones(n) / n
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if metodo == "target" and objetivo is not None:
        restricciones.append({"type": "eq", "fun": lambda w: np.dot(w, media) - objetivo})
        objetivo_funcion = riesgo
    elif metodo == "sharpe":
        objetivo_funcion = sharpe
    else:
        objetivo_funcion = riesgo

    limites = [(-1, 1) for _ in range(n)]
    resultado = minimize(objetivo_funcion, w_inicial, constraints=restricciones, bounds=limites)
    return np.array(resultado.x).flatten()


def black_litterman_optimizar(retornos, P, Q, tau=0.05, metodo="sharpe"):
    media = retornos.mean().values
    cov = retornos.cov().values
    # omega más grande = más confianza en los views del usuario
    omega = np.diag([0.01] * P.shape[0])

    # Calcular media ajustada por Black-Litterman
    M = np.linalg.inv(
        np.linalg.inv(tau * cov)
        + np.dot(np.dot(P.T, np.linalg.inv(omega)), P)
    )
    ajustada_media = np.dot(
        M,
        np.dot(np.linalg.inv(tau * cov), media)
        + np.dot(np.dot(P.T, np.linalg.inv(omega)), Q),
    )

    # Optimizar maximizando Sharpe con la media ajustada
    n = len(ajustada_media)
    w_inicial = np.ones(n) / n
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    limites = [(-1, 1) for _ in range(n)]

    def sharpe_bl(w):
        ret = np.dot(w, ajustada_media)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(ret / vol) if vol != 0 else 0

    resultado = minimize(sharpe_bl, w_inicial, constraints=restricciones, bounds=limites)
    return np.array(resultado.x).flatten()


def portfolio_performance(weights, mean_returns, cov_matrix, risk_aversion_lambda):
    returns = np.sum(weights * mean_returns) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std_dev, returns


def calcular_frontera_eficiente(retornos, num_puntos=100):
    medias = retornos.mean()
    covarianza = retornos.cov()
    n_activos = len(medias)
    resultados = {"port_rets": [], "port_vols": [], "sharpe_ratio": [], "weights": []}

    for _ in range(num_puntos):
        pesos = np.random.random(n_activos)
        pesos /= np.sum(pesos)
        rendimiento = np.dot(pesos, medias)
        riesgo = np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
        sharpe = rendimiento / riesgo if riesgo != 0 else 0
        resultados["port_rets"].append(rendimiento)
        resultados["port_vols"].append(riesgo)
        resultados["sharpe_ratio"].append(sharpe)
        resultados["weights"].append(pesos)

    return pd.DataFrame(resultados)


# --- Sidebar: controles del usuario ---
st.sidebar.title("Configuración")

tickers_seleccionados = st.sidebar.multiselect(
    "Selecciona los ETFs",
    options=list(tickers_info.keys()),
    default=["TLT", "EMB", "SPY", "VWO", "GLD"],
    help="Elige al menos 2 ETFs para el análisis.",
)

if len(tickers_seleccionados) < 2:
    st.sidebar.error("Selecciona al menos 2 ETFs.")
    st.stop()

fecha_inicio = st.sidebar.date_input(
    "Fecha de inicio del análisis",
    value=date(2010, 1, 1),
    min_value=date(2000, 1, 1),
    max_value=date(2020, 1, 1),
)

fecha_fin_entrenamiento = st.sidebar.date_input(
    "Fecha de fin del entrenamiento",
    value=date(2020, 1, 1),
    min_value=date(2005, 1, 1),
    max_value=date(2022, 1, 1),
)

fecha_inicio_backtest = st.sidebar.date_input(
    "Inicio del backtesting",
    value=date(2021, 1, 1),
    min_value=date(2010, 1, 1),
    max_value=date(2023, 1, 1),
)

fecha_fin_backtest = st.sidebar.date_input(
    "Fin del backtesting",
    value=date(2025, 1, 1),
    min_value=date(2011, 1, 1),
    max_value=date.today(),
)

if fecha_fin_entrenamiento <= fecha_inicio:
    st.sidebar.error("La fecha de fin del entrenamiento debe ser posterior al inicio.")
    st.stop()

if fecha_fin_backtest <= fecha_inicio_backtest:
    st.sidebar.error("La fecha de fin del backtesting debe ser posterior al inicio.")
    st.stop()

tickers = {k: tickers_info[k] for k in tickers_seleccionados}

inicio_str = fecha_inicio.strftime("%Y-%m-%d")
fin_entrenamiento_str = fecha_fin_entrenamiento.strftime("%Y-%m-%d")
inicio_backtest_str = fecha_inicio_backtest.strftime("%Y-%m-%d")
fin_backtest_str = fecha_fin_backtest.strftime("%Y-%m-%d")
hoy_str = datetime.today().strftime("%Y-%m-%d")

# --- App principal ---
st.title("Proyecto de Optimización de Portafolios")

tabs = st.tabs(
    [
        "Introducción",
        "Selección de ETF's",
        "Estadísticas de los ETF's",
        "Portafolios Óptimos",
        "Backtesting",
        "Modelo de Black-Litterman",
    ]
)

# --- Introducción ---
with tabs[0]:
    st.header("Introducción")
    st.write(
        """
    Este proyecto tiene como objetivo analizar y optimizar un portafolio utilizando ETFs en diferentes clases de activos,
    tales como renta fija, renta variable, y materias primas. A lo largo del proyecto, se evaluará el rendimiento de estos
    activos a través de diversas métricas financieras y técnicas de optimización de portafolios, como la optimización de
    mínima volatilidad y la maximización del Sharpe Ratio.

    Usa el panel izquierdo para seleccionar los ETFs y el rango de fechas que deseas analizar.
    """
    )
    st.info(f"**ETFs seleccionados:** {', '.join(tickers_seleccionados)}  \n"
            f"**Período de análisis:** {inicio_str} → {fin_entrenamiento_str}  \n"
            f"**Período de backtesting:** {inicio_backtest_str} → {fin_backtest_str}")

# --- Selección de ETF's ---
with tabs[1]:
    st.header("Selección de ETF's")
    datos_historicos = cargar_datos(list(tickers.keys()), inicio_str, hoy_str)

    etf_caracteristicas = pd.DataFrame(
        {
            "Ticker": list(tickers.keys()),
            "Nombre": [info["nombre"] for info in tickers.values()],
            "Sector": [info["sector"] for info in tickers.values()],
            "Categoría": [info["categoria"] for info in tickers.values()],
            "Exposición": [info["exposicion"] for info in tickers.values()],
            "Moneda": [info["moneda"] for info in tickers.values()],
            "Beta": [info["beta"] for info in tickers.values()],
            "Gastos": [info["gastos"] for info in tickers.values()],
            "Duración": [info["duracion"] for info in tickers.values()],
        }
    )

    st.header("Detalle Individual de ETFs")
    for ticker, info in tickers.items():
        st.subheader(f"{info.get('nombre', 'No especificado')} ({ticker})")
        st.write(f"Descripción: {info.get('descripcion', 'No especificado')}")
        st.write(f"Índice que sigue: {info.get('indice', 'No especificado')}")
        st.write(f"Exposición: {info.get('exposicion', 'No especificado')}")
        st.write(f"Exposición ganada: {info.get('exposicion_ganada', 'No especificado')}")
        holdings_str = ", ".join(
            [
                f"{h.get('symbol', 'N/A')} ({h.get('holdingPercent', 'N/A')})"
                for h in info.get("top_holdings", [])
            ]
        )
        st.write(f"Principales contribuyentes: {holdings_str}")
        st.write(f"Países donde invierte: {', '.join(info.get('pais_inversion', ['No especificado']))}")
        st.write(f"Duración: {info.get('duracion', 'No especificado')}")
        st.write(f"Estilo: {info.get('estilo', 'No especificado')}")
        st.write(f"Gastos: {info.get('gastos', 'No especificado')}")
        st.write(f"Moneda: {info.get('moneda', 'No especificado')}")
        st.write(f"Beta: {info.get('beta', 'No especificado')}")

    st.subheader("Características de los ETFs Seleccionados")
    st.dataframe(etf_caracteristicas)

    st.subheader("Series de Tiempo de los Precios de Cierre")
    for ticker, info in tickers.items():
        fig = px.line(
            x=datos_historicos[ticker].index,
            y=datos_historicos[ticker]["Close"].values.flatten(),
            title=f"Precio de Cierre - {ticker}",
        )
        st.plotly_chart(fig)

# --- Estadísticas de los ETF's ---
with tabs[2]:
    st.header(f"Estadísticas de los ETF's ({inicio_str} - {fin_entrenamiento_str})")
    datos_entrenamiento = cargar_datos(list(tickers.keys()), inicio_str, fin_entrenamiento_str)

    for ticker, descripcion in tickers.items():
        st.subheader(f"{descripcion['nombre']} ({ticker})")

        data = datos_entrenamiento[ticker].dropna()
        precios = data["Close"]
        retornos = data["Retornos"]

        media = retornos.mean() * 100
        volatilidad = retornos.std() * 100
        sesgo = skew(retornos)
        curtosis_val = kurtosis(retornos)
        sharpe = media / volatilidad if volatilidad != 0 else np.nan
        sortino = (
            media / retornos[retornos < 0].std()
            if retornos[retornos < 0].std() != 0
            else np.nan
        )
        VaR_95 = np.percentile(retornos, 5)
        CVaR_95 = retornos[retornos <= VaR_95].mean()

        drawdown, watermark = calcular_drawdown_y_watermark(precios)

        st.write("### Tabla de Métricas")
        metricas = pd.DataFrame(
            {
                "Métrica": [
                    "Media (%)", "Volatilidad (%)", "Sesgo", "Curtosis",
                    "Sharpe Ratio", "Sortino Ratio", "VaR 95%", "CVaR 95%",
                ],
                "Valor": [media, volatilidad, sesgo, curtosis_val, sharpe, sortino, VaR_95, CVaR_95],
            }
        )
        st.dataframe(metricas)

        st.write("### Rendimientos Acumulados")
        fig_rendimientos = graficar_linea(
            x_column=data.index,
            y_column=(1 + retornos).cumprod(),
            title=f"Rendimientos Acumulados - {descripcion['nombre']}",
            labels={"x": "Fecha", "y": "Rendimientos Acumulados"},
        )
        st.plotly_chart(fig_rendimientos)

        st.write("### Distribución de Retornos")
        fig_dist = px.histogram(
            retornos,
            nbins=50,
            title="Distribución de Retornos",
            labels={"value": "Retornos", "index": "Frecuencia"},
        )
        fig_dist.add_vline(x=VaR_95, line_dash="dash", line_color="red", annotation_text="VaR 95%", annotation_position="top left")
        fig_dist.add_vline(x=CVaR_95, line_dash="dot", line_color="orange", annotation_text="CVaR 95%", annotation_position="top left")
        st.plotly_chart(fig_dist)

        st.write("### Serie de Tiempo del Precio con Drawdowns y Watermark")
        precios_uni = precios.values.flatten()
        drawdown_uni = drawdown.values.flatten()
        watermark_uni = watermark.values.flatten()

        if len(precios_uni) != len(drawdown_uni):
            drawdown_uni = np.resize(drawdown_uni, precios_uni.shape)

        drawdown_curve = precios_uni + (drawdown_uni * precios_uni)

        fig_drawdown = px.line(
            x=data.index,
            y=precios_uni,
            title=f"Precio del ETF - {descripcion['nombre']}",
            labels={"x": "Fecha", "y": "Precio del ETF"},
        )
        fig_drawdown.add_scatter(x=data.index, y=watermark_uni, mode="lines", name="Watermark", line=dict(color="blue", dash="dash"))
        fig_drawdown.add_scatter(x=data.index, y=drawdown_curve, mode="lines", name="Drawdown", line=dict(color="red", dash="dot"))
        st.plotly_chart(fig_drawdown)

# --- Portafolios Óptimos ---
with tabs[3]:
    st.header(f"Portafolios Óptimos ({inicio_str} - {fin_entrenamiento_str})")

    datos_2010_2020 = cargar_datos(list(tickers.keys()), inicio_str, fin_entrenamiento_str)
    retornos_2010_2020 = pd.DataFrame(
        {k: v["Retornos"] for k, v in datos_2010_2020.items()}
    ).dropna()

    st.subheader("Portafolio de Mínima Volatilidad")
    pesos_min_vol = optimizar_portafolio_markowitz(retornos_2010_2020, metodo="min_vol")
    st.write("Pesos del Portafolio de Mínima Volatilidad:")
    for ticker, peso in zip(tickers.keys(), pesos_min_vol):
        st.write(f"{ticker}: {peso:.2%}")
    fig_min_vol = px.bar(x=list(tickers.keys()), y=pesos_min_vol, title="Pesos - Mínima Volatilidad")
    st.plotly_chart(fig_min_vol)

    st.subheader("Portafolio de Máximo Sharpe Ratio")
    pesos_sharpe = optimizar_portafolio_markowitz(retornos_2010_2020, metodo="sharpe")
    st.write("Pesos del Portafolio de Máximo Sharpe Ratio:")
    for ticker, peso in zip(tickers.keys(), pesos_sharpe):
        st.write(f"{ticker}: {peso:.2%}")
    fig_sharpe = px.bar(x=list(tickers.keys()), y=pesos_sharpe, title="Pesos - Máximo Sharpe Ratio")
    st.plotly_chart(fig_sharpe)

    # Frontera eficiente con 2000 portafolios simulados
    frontera = calcular_frontera_eficiente(retornos_2010_2020, num_puntos=2000)

    # Ordenar por volatilidad para trazar la curva
    frontera_sorted = frontera.sort_values("port_vols")

    fig = go.Figure()

    # Nube de portafolios simulados coloreados por Sharpe
    fig.add_trace(go.Scatter(
        x=frontera_sorted["port_vols"],
        y=frontera_sorted["port_rets"],
        mode="markers",
        marker=dict(
            color=frontera_sorted["sharpe_ratio"],
            colorscale="Viridis",
            size=4,
            opacity=0.5,
            colorbar=dict(title="Sharpe Ratio"),
        ),
        name="Portafolios simulados",
    ))

    # Calcular volatilidad y rendimiento de cada portafolio óptimo
    media_ret = retornos_2010_2020.mean()
    cov_ret = retornos_2010_2020.cov()

    def vol_port(w): return np.sqrt(np.dot(w, np.dot(cov_ret, w)))
    def ret_port(w): return np.dot(w, media_ret)

    vol_min = vol_port(pesos_min_vol)
    ret_min = ret_port(pesos_min_vol)
    vol_sharpe = vol_port(pesos_sharpe)
    ret_sharpe = ret_port(pesos_sharpe)
    peso_eq = np.array([1 / len(tickers_seleccionados)] * len(tickers_seleccionados))
    vol_eq = vol_port(peso_eq)
    ret_eq = ret_port(peso_eq)

    # Portafolio de Mínima Volatilidad
    fig.add_trace(go.Scatter(
        x=[vol_min], y=[ret_min],
        mode="markers",
        marker=dict(color="blue", size=14, symbol="diamond"),
        name="Mínima Volatilidad",
    ))

    # Portafolio de Máximo Sharpe
    fig.add_trace(go.Scatter(
        x=[vol_sharpe], y=[ret_sharpe],
        mode="markers",
        marker=dict(color="red", size=14, symbol="star"),
        name="Máximo Sharpe Ratio",
    ))

    # Portafolio Equitativo
    fig.add_trace(go.Scatter(
        x=[vol_eq], y=[ret_eq],
        mode="markers",
        marker=dict(color="orange", size=14, symbol="square"),
        name="Equitativo",
    ))

    fig.update_layout(
        title="Frontera Eficiente Simulada",
        xaxis_title="Volatilidad",
        yaxis_title="Rendimiento Esperado",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig)

    log_ret = np.log(retornos_2010_2020 + 1).dropna()
    mean_returns = log_ret.mean()
    cov_matrix = log_ret.cov()

    constraints_target = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "eq", "fun": lambda x: np.sum(x * mean_returns) * 252 - target_return},
    )

    def min_volatility_for_target_return(weights, mean_returns, cov_matrix, target_return):
        vol, ret = portfolio_performance(weights, mean_returns, cov_matrix, risk_aversion_lambda)
        return vol

    n = len(mean_returns)
    bounds = tuple((-1, 1) for _ in range(n))
    opt_target = sco.minimize(
        min_volatility_for_target_return,
        n * [1.0 / n],
        args=(mean_returns, cov_matrix, target_return),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints_target,
    )

# --- Backtesting ---
with tabs[4]:
    st.header(f"Backtesting ({inicio_backtest_str} - {fin_backtest_str})")
    st.write(
        f"""
    En esta sección se pusieron a prueba las optimizaciones obtenidas para el periodo
    {inicio_backtest_str} a {fin_backtest_str}. Los resultados se encuentran condensados
    en las siguientes gráficas y tablas.
    """
    )

    datos_backtest = cargar_datos(list(tickers.keys()), inicio_backtest_str, fin_backtest_str)
    retornos_backtest = pd.DataFrame(
        {k: v["Retornos"] for k, v in datos_backtest.items()}
    ).dropna()

    rendimientos_acumulados = pd.DataFrame(index=retornos_backtest.index)

    sp500_data = yf.download("^GSPC", start=inicio_backtest_str, end=fin_backtest_str, auto_adjust=True)["Close"]
    sp_retornos = sp500_data.pct_change().dropna()

    n_activos = len(tickers_seleccionados)
    peso_equitativo = [1 / n_activos] * n_activos

    portafolios = [
        ("Mínima Volatilidad", pesos_min_vol),
        ("Máximo Sharpe Ratio", pesos_sharpe),
        ("Equitativo", peso_equitativo),
    ]

    metricas_final = [0, 0, 0, 0, 0, 0, 0, 0]

    for nombre, pesos in portafolios:
        ret_port = np.sum(retornos_backtest * pesos, axis=1)
        media_p = ret_port.mean() * 100
        vol_p = ret_port.std() * 100
        sesgo_p = skew(ret_port)
        curtosis_p = kurtosis(ret_port)
        sharpe_p = media_p / vol_p if vol_p != 0 else np.nan
        sortino_p = (
            media_p / ret_port[ret_port < 0].std()
            if ret_port[ret_port < 0].std() != 0
            else np.nan
        )
        VaR_p = np.percentile(ret_port, 5)
        CVaR_p = ret_port[ret_port <= VaR_p].mean()
        metricas = [media_p, vol_p, sesgo_p, curtosis_p, sharpe_p, sortino_p, VaR_p, CVaR_p]
        metricas_final = np.column_stack((metricas_final, metricas))

    sp_ret_flat = sp_retornos.values.flatten()
    sp_media = sp_ret_flat.mean() * 100
    sp_vol = sp_ret_flat.std() * 100
    sp_sesgo = skew(sp_ret_flat)
    sp_curtosis = kurtosis(sp_ret_flat)
    sp_sharpe = sp_media / sp_vol if sp_vol != 0 else np.nan
    sp_sortino = (
        sp_media / sp_ret_flat[sp_ret_flat < 0].std()
        if sp_ret_flat[sp_ret_flat < 0].std() != 0
        else np.nan
    )
    sp_var95 = np.percentile(sp_ret_flat, 5)
    sp_cvar95 = sp_ret_flat[sp_ret_flat <= sp_var95].mean()
    sp_metricas = [sp_media, sp_vol, sp_sesgo, sp_curtosis, sp_sharpe, sp_sortino, sp_var95, sp_cvar95]

    metricas_final = metricas_final[:, 1:]
    metricas_final = np.column_stack((metricas_final, sp_metricas))

    st.write(
        pd.DataFrame(
            metricas_final,
            columns=["Mínima volatilidad", "Máximo Sharpe Ratio", "Equitativo", "S&P 500"],
            index=["Media (%)", "Volatilidad (%)", "Sesgo", "Curtosis", "Sharpe Ratio", "Sortino Ratio", "VaR 95%", "CVaR 95%"],
        )
    )

    st.subheader("Rendimientos Acumulados de los Portafolios")
    for nombre, pesos in portafolios:
        pesos_arr = np.array(pesos).flatten()
        ret_serie = retornos_backtest.values.dot(pesos_arr)
        rendimientos_acumulados[nombre] = pd.Series(ret_serie, index=retornos_backtest.index).cumsum()

    sp_retornos_cumsum = pd.Series(sp_retornos.values.flatten(), index=sp_retornos.index).cumsum()

    fig_rendimientos = px.line(
        rendimientos_acumulados,
        title="Rendimientos Acumulados - Comparación de Portafolios",
        labels={"value": "Rendimientos Acumulados", "index": "Fecha"},
    )
    fig_rendimientos.add_trace(
        go.Scatter(
            x=sp_retornos_cumsum.index,
            y=sp_retornos_cumsum.values,
            mode="lines",
            name="S&P 500",
            line=dict(color="red", dash="solid"),
        )
    )
    st.plotly_chart(fig_rendimientos)

# --- Modelo de Black-Litterman ---
with tabs[5]:
    st.header("Modelo de Optimización Black-Litterman")

    n_activos = len(tickers_seleccionados)
    P = np.eye(n_activos)

    st.write("### Define tus views de rendimiento esperado (anual) para cada ETF:")
    Q_values = []
    for ticker in tickers_seleccionados:
        val = st.slider(
            f"Rendimiento esperado - {ticker}",
            min_value=-10,
            max_value=30,
            value=5,
            step=1,
            format="%d%%",
        )
        Q_values.append(val / 100)
    Q = np.array(Q_values)

    pesos_black_litterman = black_litterman_optimizar(retornos_2010_2020, P, Q)

    st.write("### Pesos del Portafolio Ajustado con Black-Litterman:")
    for ticker, peso in zip(tickers.keys(), pesos_black_litterman):
        st.write(f"{ticker}: {peso:.2%}")

    fig_black_litterman = px.bar(
        x=list(tickers.keys()),
        y=pesos_black_litterman,
        title="Pesos Ajustados - Black-Litterman",
        labels={"x": "ETF", "y": "Peso"},
    )
    st.plotly_chart(fig_black_litterman)

    st.write(
        """
    Los pesos ajustados según el Modelo de Black-Litterman reflejan un portafolio optimizado que combina
    tus views con la distribución a priori de los activos. Ajusta los sliders de rendimiento esperado
    para ver cómo cambian los pesos del portafolio.
    """
    )

    # --- Backtesting de Black-Litterman vs otros portafolios ---
    st.write("---")
    st.subheader(f"Backtesting Black-Litterman vs otros portafolios ({inicio_backtest_str} - {fin_backtest_str})")

    portafolios_bl = [
        ("Mínima Volatilidad", pesos_min_vol),
        ("Máximo Sharpe Ratio", pesos_sharpe),
        ("Equitativo", [1 / len(tickers_seleccionados)] * len(tickers_seleccionados)),
        ("Black-Litterman", pesos_black_litterman),
    ]

    rendimientos_bl = pd.DataFrame(index=retornos_backtest.index)
    metricas_bl = [0, 0, 0, 0, 0, 0, 0, 0]

    for nombre, pesos in portafolios_bl:
        pesos_arr = np.array(pesos).flatten()
        ret_serie = retornos_backtest.values.dot(pesos_arr)
        serie = pd.Series(ret_serie, index=retornos_backtest.index)
        rendimientos_bl[nombre] = serie.cumsum()

        media_p = serie.mean() * 100
        vol_p = serie.std() * 100
        sesgo_p = skew(serie)
        curtosis_p = kurtosis(serie)
        sharpe_p = media_p / vol_p if vol_p != 0 else np.nan
        sortino_p = (
            media_p / serie[serie < 0].std()
            if serie[serie < 0].std() != 0
            else np.nan
        )
        VaR_p = np.percentile(serie, 5)
        CVaR_p = serie[serie <= VaR_p].mean()
        metricas_bl = np.column_stack((metricas_bl, [media_p, vol_p, sesgo_p, curtosis_p, sharpe_p, sortino_p, VaR_p, CVaR_p]))

    metricas_bl = metricas_bl[:, 1:]

    st.write("### Métricas comparativas")
    st.dataframe(
        pd.DataFrame(
            metricas_bl,
            columns=["Mínima Volatilidad", "Máximo Sharpe Ratio", "Equitativo", "Black-Litterman"],
            index=["Media (%)", "Volatilidad (%)", "Sesgo", "Curtosis", "Sharpe Ratio", "Sortino Ratio", "VaR 95%", "CVaR 95%"],
        )
    )

    st.write("### Rendimientos Acumulados")
    sp_bl_cumsum = pd.Series(sp_retornos.values.flatten(), index=sp_retornos.index).cumsum()

    fig_bl = px.line(
        rendimientos_bl,
        title="Rendimientos Acumulados - Black-Litterman vs otros portafolios",
        labels={"value": "Rendimientos Acumulados", "variable": "Portafolio", "index": "Fecha"},
    )
    fig_bl.add_trace(
        go.Scatter(
            x=sp_bl_cumsum.index,
            y=sp_bl_cumsum.values,
            mode="lines",
            name="S&P 500",
            line=dict(color="black", dash="dot"),
        )
    )
    st.plotly_chart(fig_bl)
