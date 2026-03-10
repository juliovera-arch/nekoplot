import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Configuración de página
st.set_page_config(page_title="LabPlot Pro - Análisis Físico", layout="wide")

# --- FUNCIONES DE AJUSTE ---
def modelos(x, tipo):
    if tipo == "Lineal (y = mx + b)":
        return lambda x, m, b: m * x + b
    elif tipo == "Proporcional (y = mx)":
        return lambda x, m: m * x
    elif tipo == "Potencial (y = ax^b)":
        return lambda x, a, b: a * np.power(x, b)
    elif tipo == "Exponencial (y = a*e^{bx})":
        return lambda x, a, b: a * np.exp(b * x)
    elif tipo == "Inversa (y = a/x)":
        return lambda x, a: a / x
    elif tipo == "Cuadrática (y = ax^2 + bx + c)":
        return lambda x, a, b, c: a * x**2 + b * x + c
    elif tipo == "Seno (y = a*sin(bx+c)+d)":
        return lambda x, a, b, c, d: a * np.sin(b * x + c) + d

# --- INTERFAZ LATERAL ---
st.sidebar.title("📊 Configuración de Datos")
metodo_entrada = st.sidebar.radio("Entrada de datos:", ["Manual", "Subir CSV"])

if metodo_entrada == "Manual":
    df_init = pd.DataFrame(
        [[1.0, 2.0, 0.1, 0.2], [2.0, 3.9, 0.1, 0.2], [3.0, 6.1, 0.1, 0.2]],
        columns=['x', 'y', 'dx', 'dy']
    )
    df = st.sidebar.data_editor(df_init, num_rows="dynamic")
else:
    archivo = st.sidebar.file_uploader("Cargar CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
    else:
        st.info("Sube un archivo para comenzar.")
        st.stop()

# --- OPCIONES DE AJUSTE ---
st.sidebar.divider()
tipo_ajuste = st.sidebar.selectbox("Tipo de Ajuste", [
    "Lineal (y = mx + b)", "Proporcional (y = mx)", "Potencial (y = ax^b)", 
    "Exponencial (y = a*e^{bx})", "Inversa (y = a/x)", "Cuadrática (y = ax^2 + bx + c)", "Seno (y = a*sin(bx+c)+d)"
])
show_errors = st.sidebar.checkbox("Mostrar barras de error", value=True)

# --- PROCESAMIENTO ---
st.title("📈 LabPlot Pro")
st.markdown("Analizador de datos científicos con incertidumbres.")

try:
    x = df['x'].values
    y = df['y'].values
    dx = df['dx'].values if 'dx' in df.columns else None
    dy = df['dy'].values if 'dy' in df.columns else None

    # Ajuste de curva
    func = modelos(x, tipo_ajuste)
    popt, pcov = curve_fit(func, x, y)
    
    # Generación de línea de ajuste
    x_range = np.linspace(min(x), max(x), 100)
    y_fit = func(x_range, *popt)
    
    # Cálculo de R²
    y_pred = func(x, *popt)
    r2 = r2_score(y, y_pred)

    # --- GRÁFICO CON PLOTLY (VIRIDIS) ---
    fig = go.Figure()

    # Puntos de datos
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Datos experimentales',
        error_x=dict(type='data', array=dx, visible=show_errors),
        error_y=dict(type='data', array=dy, visible=show_errors),
        marker=dict(color='#440154', size=10) # Parte del esquema Viridis
    ))

    # Línea de ajuste
    fig.add_trace(go.Scatter(
        x=x_range, y=y_fit,
        mode='lines',
        name='Ajuste',
        line=dict(color='#21918c', width=3) # Parte del esquema Viridis
    ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Variable Independiente (x)",
        yaxis_title="Variable Dependiente (y)",
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- RESULTADOS ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📝 Ecuación del Ajuste")
        params_str = ", ".join([f"{p:.4f}" for p in popt])
        st.latex(rf"R^2 = {r2:.4f}")
        st.info(f"Parámetros optimizados: {params_str}")
    
    with col2:
        st.subheader("📥 Exportar")
        # El gráfico de Plotly permite descargar en SVG/PNG desde el menú flotante
        st.write("Puedes descargar el gráfico como vector (SVG) usando la cámara en la esquina superior derecha del gráfico.")

except Exception as e:
    st.error(f"Error en el ajuste: Asegúrate de que los datos sean compatibles con el modelo seleccionado. Detalle: {e}")
