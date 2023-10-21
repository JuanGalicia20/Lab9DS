import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit.components.v1 as components
from streamlit_extras.metric_cards import style_metric_cards
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet



st.markdown("""
<style>
body {
  background: #ff0099; 
  background: -webkit-linear-gradient(to right, #ff0099, #493240); 
  background: linear-gradient(to right, #ff0099, #493240); 
}
</style>
    """, unsafe_allow_html=True)

# Configurar el título y el icono de la pestaña del navegador
logo_container = st.sidebar.container()

# Añadir el logo al contenedor del logo
logo_container.image("truck.png", width=60)


# Create a custom layout for the rectangle
rect_html = """
<div style="background-color: #3498db; padding: 10px; border-radius: 10px; margin: 0; width: 100%;">
    <h4 style="color: white; text-align: left;">Hola, Matthew</h4>
    <h7 style="color: white; text-align: left;">¡Precios y costos de la empresa para el día de Hoy!"</h7>
</div>
"""

# Add the rectangle to the Streamlit page
st.markdown(rect_html, unsafe_allow_html=True)

# Barra lateral en la parte izquierda
st.sidebar.header("Opciones")


data = pd.read_csv("precios.csv")

data["Fecha"] = pd.to_datetime(data["Fecha"])

data.set_index(data["Fecha"], inplace=True, drop=True)
data.drop(columns=["Fecha"], inplace=True)

data['Precio'] = data['Precio'].interpolate()


# Agregar elementos a la barra lateral
option = st.sidebar.selectbox("Selecciona una opción", ("Modelo 1", "Modelo 2", "Modelo 3"))

promedio = round(data["Precio"].mean(),2)
prom = "Q" + str(promedio)


#Modelo SARIMAX
model1p = SARIMAX(data, order=(1,1,1), seasonal_order=(5,1,0,12))
resultado1p = model1p.fit()
pred1p = resultado1p.get_forecast(steps=90)
pred1pvalues = pred1p.predicted_mean
pred1pc = pred1p.conf_int()

fecha_inicio = data.index.min()
fecha_fin = data.index.max()

# Genera un rango de fechas consecutivas para las predicciones
prediccion_index = pd.date_range(start=fecha_fin + pd.DateOffset(months=1), periods=90)

c2023 = data["2023"]["Precio"].sum()

preds = resultado1p.get_forecast(steps=1)
print(preds)
# Contenido principal del dashboard
if option == "Modelo 1":
    cola1, cola2, cola3 = st.columns(3)
    cola1.metric(label="Costos 2023", value="Q"+str(round(c2023,2)), delta="+15.6%")
    cola2.metric(label="Precio Predecido de Mañana", value="Q"+str(round(pred1pvalues.iloc[0],2)), delta="+12.7%")
    cola3.metric(label="Precio Promedio", value=prom, delta="-12.7%")
    style_metric_cards()
    col1, col2 = st.columns(2)

    # Gráfico 1 en la primera columna
    with col1:
        st.subheader("Serie de tiempo")
        fig1, ax1 = plt.subplots()
        sns.lineplot(data=data, x=data.index, y="Precio", color="#4877B7", ax=ax1)
        st.pyplot(fig1)

    # Gráfico 2 en la segunda columna
    with col2:
        fig, ax = plt.subplots()

        # Crear el histograma en el eje usando Seaborn
        sns.histplot(data, kde=False, color='#48AEB7', ax=ax)

        # Mostrar el histograma en Streamlit
        st.subheader("Frecuencia de precios")
        st.pyplot(fig)
        
    c1 = st.columns(1)
    st.subheader("Predicción")
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=data, x=data.index, y="Precio", color="#4877B7", ax=ax2)
    sns.lineplot(data=pred1pvalues, x=prediccion_index, y=pred1pvalues, ax=ax2, color="#48b789")
    plt.fill_between(prediccion_index, pred1pc["lower Precio"], pred1pc["upper Precio"], color="pink")
    st.pyplot(fig2)

elif option == "Modelo 2":
    cola1, cola2, cola3 = st.columns(3)
    cola1.metric(label="Costos 2023", value="Q9,328.55", delta="+15.6%")
    cola2.metric(label="Precio del dia", value="Q3.01", delta="+12.7%")
    cola3.metric(label="Precio Promedio", value=prom, delta="-12.7%")
    style_metric_cards()
    col1, col2 = st.columns(2)

    # Gráfico 1 en la primera columna
    with col1:
        st.subheader("Serie de tiempo")
        fig1, ax1 = plt.subplots()
        sns.lineplot(data=data, x=data.index, y="Precio", color="#4877B7", ax=ax1)
        st.pyplot(fig1)

    # Gráfico 2 en la segunda columna
    with col2:
        fig, ax = plt.subplots()

        # Crear el histograma en el eje usando Seaborn
        sns.histplot(data, kde=False, color='#48AEB7', ax=ax)

        # Mostrar el histograma en Streamlit
        st.subheader("Frecuencia de precios")
        st.pyplot(fig)
        
    c1 = st.columns(1)
    st.subheader("Predicción Modelo EMA")
    span = 3  # Parámetro de suavizado exponencial
    ema = data['Precio'].ewm(span=span, adjust=False).mean()
    future_dates = pd.date_range(start=data.index[-1], periods=30 + 1, closed='right')
    future_ema = ema[-1]  # El último valor EMA conocido
    
    for date in future_dates:
        future_ema = future_ema + (ema[-1] - future_ema) / span
        ema = ema.append(pd.Series(future_ema, index=[date]))
    fig3, ax3 = plt.subplots()
    sns.lineplot(data=data, x=data.index, y="Precio", color="#4877B7", ax=ax3)
    sns.lineplot(data=ema, x=ema.index, y=ema, color="#48b789", ax=ax3)
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.title('Predicción de Serie de Tiempo con EMA')
    plt.grid(True)
    st.pyplot(fig3)
    
    
    
else:
    data2 = data.copy()
    data2.reset_index(inplace=True)
    data2.rename(columns={"Fecha":"ds", "Precio":"y"}, inplace=True)
    m = Prophet()
    m.fit(data2)
    futurep = m.make_future_dataframe(periods = 2160)
    forecastp = m.predict(futurep)
    
    cola1, cola2, cola3 = st.columns(3)
    cola1.metric(label="Costos 2023", value="Q9,328.55", delta="+15.6%")
    cola2.metric(label="Precio del dia", value="Q"+str(round(forecastp.iloc[-1,1],2)), delta="+12.7%")
    cola3.metric(label="Precio Promedio", value=prom, delta="-12.7%")
    style_metric_cards()
    col1, col2 = st.columns(2)

    # Gráfico 1 en la primera columna
    with col1:
        st.subheader("Serie de tiempo")
        fig1, ax1 = plt.subplots()
        sns.lineplot(data=data, x=data.index, y="Precio", color="#4877B7", ax=ax1)
        st.pyplot(fig1)

    # Gráfico 2 en la segunda columna
    with col2:
        fig, ax = plt.subplots()

        # Crear el histograma en el eje usando Seaborn
        sns.histplot(data, kde=False, color='#48AEB7', ax=ax)

        # Mostrar el histograma en Streamlit
        st.subheader("Frecuencia de precios")
        st.pyplot(fig)
        
    c1 = st.columns(1)
    st.subheader("Predicción Modelo Prophet")
    
    fig4, ax4 = plt.subplots()
    st.subheader("Predicción Prophet")
    sns.lineplot(data=data, x=data.index, y="Precio", color="#4877B7", ax=ax4, label="Historico")
    sns.lineplot(data = forecastp, x="ds",y="yhat", color="#48b789", ax=ax4, label="Prediccion")
    plt.fill_between(forecastp["ds"], forecastp["yhat_lower"], forecastp["yhat_upper"], color="pink",  label="Error Prediccion")
    plt.legend()
    st.pyplot(fig4)




























