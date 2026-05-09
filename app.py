import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuración de la página
st.set_page_config(page_title="CardioPredict", page_icon="🫀", layout="centered")

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    with open('consenso_modelos.pkl', 'rb') as archivo:
        return pickle.load(archivo)

try:
    datos_guardados = cargar_modelo()
    modelos = datos_guardados['modelos']
except FileNotFoundError:
    st.error("No se encontró el archivo consenso_modelos.pkl")
    st.stop()

# Encabezado
st.image("https://images.unsplash.com/photo-1551076805-e1869033e561?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", use_container_width=True)
st.title("🫀 CardioPredict CDSS")
st.markdown("### Consenso Multi-Modelo con Inteligencia Artificial")

# Formulario de entrada
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Edad (Años)", min_value=18, max_value=120, value=60)
    ef = st.number_input("Fracción de Eyección (FEVI %)", min_value=10, max_value=80, value=35)
with col2:
    creatinine = st.number_input("Creatinina Sérica (mg/dL)", min_value=0.1, max_value=15.0, value=1.2, step=0.1)
    sodium = st.number_input("Sodio Sérico (mEq/L)", min_value=110, max_value=160, value=135)

if st.button("Analizar Paciente", type="primary", use_container_width=True):
    entrada = np.array([[age, ef, creatinine, sodium]])
    
    resultados =[]
    suma_prob = 0
    
    for nombre, modelo in modelos.items():
        prob = modelo.predict_proba(entrada)[0][1] * 100
        resultados.append({"Modelo de IA": nombre, "Predicción (%)": round(prob, 1)})
        suma_prob += prob
        
    prob_final = round(suma_prob / len(modelos), 1)
    
    hallazgos =[]
    if ef < 40: hallazgos.append("Fracción de eyección reducida.")
    if creatinine > 1.2: hallazgos.append("Creatinina sérica elevada.")
    if sodium < 135: hallazgos.append("Hiponatremia detectada.")
    if not hallazgos: hallazgos.append("Biomarcadores estables.")
    
    st.divider()
    
    if prob_final >= 60:
        st.error(f"🚨 Diagnóstico: Riesgo Crítico (Mortalidad Estimada: {prob_final}%)")
        st.warning("Recomendación: Ingreso a UCI o evaluación cardiológica urgente.")
    elif prob_final >= 35:
        st.warning(f"⚠️ Diagnóstico: Riesgo Moderado (Mortalidad Estimada: {prob_final}%)")
        st.info("Recomendación: Programar ecocardiograma y control.")
    else:
        st.success(f"✅ Diagnóstico: Riesgo Bajo (Mortalidad Estimada: {prob_final}%)")
        st.info("Recomendación: Manejo ambulatorio.")
        
    st.markdown("#### Métricas Individuales de los Modelos")
    st.table(pd.DataFrame(resultados))
    
    st.markdown("#### Hallazgos Clínicos")
    for h in hallazgos:
        st.markdown(f"- {h}")

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
    <b>Proyecto de Machine Learning - Ingeniería Biomédica</b><br>
    Equipo: Yarod Jhair Quiroga Jaimes, Alejandra Cortez Ramos, Maria Paula Lamboglia, Sara Saavedra
</div>
""", unsafe_allow_html=True)