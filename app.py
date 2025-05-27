import streamlit as st
import joblib
import numpy as np

st.title("Predicción con modelo de regresión lineal")
st.write("Este modelo fue entrenado con datos simples: y = 2 * x")

modelo = joblib.load("modelo.pkl")

x = st.number_input("Ingrese un valor de x", value=0)

if st.button("Predecir"):
    pred = modelo.predict(np.array([[x]]))[0]
    st.success(f"La predicción es: {pred:.2f}")
