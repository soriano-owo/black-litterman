import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from scipy.optimize import minimize
import scipy.optimize as sco
from scipy.stats import skew, kurtosis

# --- Configuración de los ETFs ---
tickers = {
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
        "descripcion": "Este ETF sigue el índice FTSE Emerging Markets All Cap China A Inclusion Index, que incluye acciones de mercados emergentes en Asia, Europa, América Latina y África.",
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
}

# --- Funciones Auxiliares ---
def cargar_datos(tickers_list, inicio, fin):
    """Descarga datos históricos para una lista de tickers desde Yahoo Finance."""
    datos = {}
    for ticker in tickers_list:
        df = yf.download(ticker, start=inicio, end=fin)
        df["Retornos"] = df["Close"].pct_change()
        datos[ticker] = df
    return datos


def graficar_linea(x_column, y_column, title, labels=None):
    """Crea un gráfico de línea con arrays directos."""
    if isinstance(y_column, pd.Series):
        y_column = y_column.values.flatten()
    fig = px.line(x=x_column, y=y_column, title=title, labels=labels)
    return fig


def calcular_drawdown_y_watermark(precios):
    """Calcula el drawdown y el watermark basado en precios."""
    watermark = precios.cummax()
    drawdown = (precios / watermark) - 1
    return drawdown, watermark


# Medida de aversión al riesgo
risk_aversion_lambda = 1

# Target de retorno anual
target_return = 0.10


def optimizar_portafolio_markowitz(retornos, metodo="min_vol", objetivo=None):
    """Optimiza el portafolio según el modelo de Markowitz."""
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


def black_litterman_optimizar(retornos, P, Q, tau=0.05, metodo="min_vol"):
    """Optimiza el portafolio utilizando el modelo de Black-Litterman."""
    media = retornos.mean()
    cov = retornos.cov()

    M = np.linalg.inv(
        np.linalg.inv(tau * cov)
        + np.dot(np.dot(P.T, np.linalg.inv(np.diag([1] * P.shape[0]))), P)
    )
    ajustada_media = np.dot(
        M,
        np.dot(np.linalg.inv(tau * cov), media)
        + np.dot(np.dot(P.T, np.linalg.inv(np.diag([1] * P.shape[0]))), Q),
    )
    return optimizar_portafolio_markowitz(retornos, metodo=metodo)


def portfolio_performance(weights, mean_returns, cov_matrix, risk_aversion_lambda):
    """Obtiene el performance del portafolio."""
    returns = np.sum(weights * mean_returns) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std_dev, returns


def calcular_frontera_eficiente(retornos, num_puntos=100):
    """Calcula la frontera eficiente generando múltiples portafolios aleatorios."""
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


# --- Configuración de Streamlit ---
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
    """
    )

# --- Selección de ETF's ---
with tabs[1]:
    st.header("Selección de ETF's")
    hoy = datetime.today().strftime("%Y-%m-%d")
    datos_2010_hoy = cargar_datos(list(tickers.keys()), "2010-01-01", hoy)

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
            "Rango 1 Año": [info["rango_1y"] for info in tickers.values()],
            "Rendimiento YTD": [info["rendimiento_ytd"] for info in tickers.values()],
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
        st.write(f"Moneda de denominación: {info.get('moneda', 'No especificado')}")
        st.write(f"Beta: {info.get('beta', 'No especificado')}")
        st.write(f"Rango en el último año: {info.get('rango_1y', 'No especificado')}")
        st.write(f"Rendimiento YTD: {info.get('rendimiento_ytd', 'No especificado')}")

    st.subheader("Características de los ETFs Seleccionados")
    st.dataframe(etf_caracteristicas)

    st.subheader("Series de Tiempo de los Precios de Cierre")
    for ticker, info in tickers.items():
        fig = px.line(
            x=datos_2010_hoy[ticker].index,
            y=datos_2010_hoy[ticker]["Close"].values.flatten(),
            title=f"Precio de Cierre - {ticker}",
        )
        st.plotly_chart(fig)

# --- Estadísticas de los ETF's ---
with tabs[2]:
    st.header("Estadísticas de los ETF's (2010-2023)")
    datos_2010_2023 = cargar_datos(list(tickers.keys()), "2010-01-01", "2023-01-01")

    for ticker, descripcion in tickers.items():
        st.subheader(f"{descripcion['nombre']} ({ticker})")

        data = datos_2010_2023[ticker].dropna()
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
    st.header("Portafolios Óptimos (2010-2020)")

    datos_2010_2020 = cargar_datos(list(tickers.keys()), "2010-01-01", "2020-01-01")
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

    frontera = calcular_frontera_eficiente(retornos_2010_2020)
    fig = px.scatter(
        frontera,
        x="port_vols",
        y="port_rets",
        color="sharpe_ratio",
        labels={"port_vols": "Volatilidad Esperada", "port_rets": "Rendimiento Esperado", "sharpe_ratio": "Sharpe Ratio"},
        title="Frontera Eficiente Simulada",
    ).update_traces(mode="markers", marker=dict(symbol="cross"))

    max_sharpe_idx = frontera["sharpe_ratio"].idxmax()
    fig.add_scatter(
        mode="markers",
        x=[frontera.loc[max_sharpe_idx, "port_vols"]],
        y=[frontera.loc[max_sharpe_idx, "port_rets"]],
        marker=dict(color="RoyalBlue", size=15, symbol="star"),
        name="Máximo Sharpe",
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
    st.header("Backtesting (2021-2025)")
    st.write(
        """
    En esta sección se pusieron a prueba las optimizaciones obtenidas. Para ello se empleó la técnica de backtesting,
    en la cual las estrategias fueron implementadas para el periodo de 2021 a 2025. Los resultados se encuentran condensados
    en las siguientes gráficas y tablas.
    """
    )

    datos_2021_2025 = cargar_datos(list(tickers.keys()), "2021-01-01", "2025-01-01")
    retornos_2021_2025 = pd.DataFrame(
        {k: v["Retornos"] for k, v in datos_2021_2025.items()}
    ).dropna()

    rendimientos_acumulados = pd.DataFrame(index=retornos_2021_2025.index)

    sp500_data = yf.download("^GSPC", start="2021-01-01", end="2025-01-01")["Close"]
    sp_retornos = sp500_data.pct_change().dropna()

    portafolios = [
        ("Mínima Volatilidad", pesos_min_vol),
        ("Máximo Sharpe Ratio", pesos_sharpe),
        ("Equitativo", [0.2, 0.2, 0.2, 0.2, 0.2]),
    ]

    metricas_final = [0, 0, 0, 0, 0, 0, 0, 0]

    for nombre, pesos in portafolios:
        ret_port = np.sum(retornos_2021_2025 * pesos, axis=1)
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

    # Métricas S&P 500
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
            columns=["Mínima volatilidad", "Máximo sharp ratio", "Equitativo", "S&P 500"],
            index=["Media (%)", "Volatilidad (%)", "Sesgo", "Curtosis", "Sharpe Ratio", "Sortino Ratio", "VaR 95%", "CVaR 95%"],
        )
    )

    st.subheader("Rendimientos Acumulados de los Portafolios")
    for nombre, pesos in portafolios:
        pesos_reshaped = np.array(pesos).reshape(-1, 1)
        rendimientos = retornos_2021_2025.dot(pesos_reshaped)
        rendimientos_acumulados[nombre] = rendimientos.cumsum()
        st.write(f"Rendimientos Acumulados - {nombre}")
        st.line_chart(rendimientos.cumsum())

    st.write("Rendimientos Acumulados S&P 500")
    st.line_chart(sp_retornos.cumsum())

    sp_retornos_cumsum = sp_retornos.cumsum()

    fig_rendimientos = px.line(
        rendimientos_acumulados,
        title="Rendimientos Acumulados - Comparación de Portafolios",
        labels={"value": "Rendimientos Acumulados", "index": "Fecha"},
    )
    fig_rendimientos.add_trace(
        go.Scatter(
            x=sp_retornos_cumsum.index,
            y=sp_retornos_cumsum.values.flatten(),
            mode="lines",
            name="S&P 500",
            line=dict(color="red", dash="solid"),
        )
    )
    st.plotly_chart(fig_rendimientos)

# --- Modelo de Black-Litterman ---
with tabs[5]:
    st.header("Modelo de Optimización Black-Litterman")
    P = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    Q = np.array([0.03, 0.06, 0.08, 0.11, 0.04])
    pesos_black_litterman = black_litterman_optimizar(retornos_2010_2020, P, Q)
    st.write("Pesos del Portafolio Ajustado con el Modelo de Black-Litterman:")
    for ticker, peso in zip(tickers.keys(), pesos_black_litterman):
        st.write(f"{ticker}: {peso:.2%}")
    fig_black_litterman = px.bar(
        x=list(tickers.keys()),
        y=pesos_black_litterman,
        title="Pesos Ajustados - Black-Litterman",
    )
    st.plotly_chart(fig_black_litterman)

    st.write(
        """
    Los pesos ajustados según el Modelo de Black-Litterman reflejan un portafolio optimizado que combina tus views
    con la distribución a priori de los activos. A continuación, se presenta un análisis detallado de los resultados
    y su interpretación.

    **TLT (21.03%)**: El modelo asigna un peso significativo a los bonos de largo plazo, destacando su estabilidad
    en escenarios de aversión al riesgo.

    **EMB (52.25%)**: La gran exposición a bonos emergentes se alinea con tu view positiva (6%) hacia este segmento.

    **SPY (36.20%)**: El peso asignado a acciones estadounidenses refleja confianza en el crecimiento económico en EE. UU.

    **VWO (-18.09%)**: El peso negativo (corto) indica que el modelo percibe un alto nivel de incertidumbre en estos activos.

    **GLD (8.60%)**: El modelo asigna una exposición moderada al oro, coherente con tu view de rendimiento estable (4%).
    """
    )
