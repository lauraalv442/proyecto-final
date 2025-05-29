import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Título
st.set_page_config(page_title="Predicción de Precios de Coches Audi", layout="centered")
st.title("🚗 Predicción de Precios y Clasificación de Coches Audi")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("audi_dataset.csv")
    df.columns = df.columns.str.lower()
    df.rename(columns={
        "model": "modelo",
        "year": "anio",
        "transmission": "transmision",
        "distance": "kilometraje",
        "fueltype": "combustible",
        "tax": "impuesto",
        "mpg": "consumo_mpg",
        "enginesize": "tamano_motor",
        "europrice": "precio"
    }, inplace=True)
    return df

df = cargar_datos()

# Preprocesamiento para regresión
def preparar_datos_regresion(df):
    df_reg = df.copy()
    for col in ['modelo', 'transmision', 'combustible']:
        df_reg[col] = LabelEncoder().fit_transform(df_reg[col])
    X = df_reg.drop("precio", axis=1)
    y = df_reg["precio"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento para clasificación
def preparar_datos_clasificacion(df):
    df_clf = df.copy()
    def clasificar_precio(p):
        if p < 12000:
            return "bajo"
        elif p < 18000:
            return "medio"
        else:
            return "alto"
    df_clf["categoria_precio"] = df_clf["precio"].apply(clasificar_precio)
    df_clf = pd.get_dummies(df_clf, columns=["modelo", "transmision", "combustible"], drop_first=True)
    X = df_clf.drop(["precio", "categoria_precio"], axis=1)
    y = df_clf["categoria_precio"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento de modelos
Xr_train, Xr_test, yr_train, yr_test = preparar_datos_regresion(df)
modelo_regresion = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_regresion.fit(Xr_train, yr_train)

Xc_train, Xc_test, yc_train, yc_test = preparar_datos_clasificacion(df)
modelo_clasificacion = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_clasificacion.fit(Xc_train, yc_train)

# Sidebar - entrada de datos
st.sidebar.title("🛠️ Configuración del Coche")

anio = st.sidebar.slider("Año", 2000, 2023, 2015)
kilometraje = st.sidebar.slider("Kilometraje (km)", 0, 200000, 60000)
impuesto = st.sidebar.slider("Impuesto anual (€)", 0, 600, 150)
consumo = st.sidebar.slider("Consumo (mpg)", 20.0, 90.0, 50.0)
tamano_motor = st.sidebar.slider("Tamaño del motor (L)", 0.8, 5.0, 2.0)

modelo = st.sidebar.selectbox("Modelo", sorted(df["modelo"].unique()))
transmision = st.sidebar.selectbox("Transmisión", sorted(df["transmision"].unique()))
combustible = st.sidebar.selectbox("Combustible", sorted(df["combustible"].unique()))

# Entrada para regresión
le_modelo = LabelEncoder().fit(df["modelo"])
le_transmision = LabelEncoder().fit(df["transmision"])
le_combustible = LabelEncoder().fit(df["combustible"])

entrada_reg = pd.DataFrame([{
    "modelo": le_modelo.transform([modelo])[0],
    "anio": anio,
    "transmision": le_transmision.transform([transmision])[0],
    "kilometraje": kilometraje,
    "combustible": le_combustible.transform([combustible])[0],
    "impuesto": impuesto,
    "consumo_mpg": consumo,
    "tamano_motor": tamano_motor
}])

# Entrada para clasificación
entrada_clf = pd.DataFrame([{
    "anio": anio,
    "kilometraje": kilometraje,
    "impuesto": impuesto,
    "consumo_mpg": consumo,
    "tamano_motor": tamano_motor
}])

# Dummy variables para clasificación
df_dummy = pd.get_dummies(df[["modelo", "transmision", "combustible"]], drop_first=True)
entrada_dummy = pd.get_dummies(pd.DataFrame([{
    "modelo": modelo,
    "transmision": transmision,
    "combustible": combustible
}]), drop_first=True)
entrada_dummy = entrada_dummy.reindex(columns=df_dummy.columns, fill_value=0)
entrada_clf = pd.concat([entrada_clf, entrada_dummy], axis=1)

# Predicción
if st.button("🔮 Predecir"):
    precio_estimado = modelo_regresion.predict(entrada_reg)[0]
    categoria = modelo_clasificacion.predict(entrada_clf)[0]

    st.subheader("📊 Resultados de la Predicción")
    st.write(f"💰 **Precio estimado del coche:** `{precio_estimado:,.2f} €`")
    st.write(f"🏷️ **Categoría de precio:** `{categoria.upper()}`")

    st.success("Predicción realizada con éxito.")
