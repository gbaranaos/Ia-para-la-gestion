"""SolarCheck AI: app educativa para clasificar estados de paneles solares."""
from pathlib import Path
from io import BytesIO
import hashlib
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from inferencia import (
    NOMBRES, leer_metadatos, cargar_modelo, clasificar,
    recomendacion_con_gemini,
)

ROOT = Path(__file__).resolve().parent
PESOS = ROOT / "modelo" / "modelo.pt"
METADATOS = ROOT / "modelo" / "clases.json"
UMBRAL_REVISION = 0.65

st.set_page_config(page_title="SolarCheck AI | Diagnóstico visual educativo",
                   page_icon="☀️", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{max-width:1260px;padding-top:1.55rem}
.hero{padding:2rem 2.2rem;border-radius:22px;background:linear-gradient(110deg,#0e3436,#145459 60%,#18766a);color:white;margin-bottom:1.25rem;box-shadow:0 11px 36px rgba(5,55,52,.13)}
.hero small{letter-spacing:.16em;font-weight:700;color:#b9efdd}
.hero h1{font-size:2.65rem;margin:.35rem 0 .55rem;color:#fff}
.hero p{color:#e2f3ed;font-size:1.08rem;margin:0;max-width:750px}
.metric-note{font-size:.78rem;color:#617879}
section[data-testid="stSidebar"]{background-color:#eef5f1}
.stButton>button[kind="primary"]{background-color:#0c7466;border-color:#0c7466}
.stButton>button[kind="primary"]:hover{background-color:#075f54;border-color:#075f54}
</style>""",unsafe_allow_html=True)

@st.cache_data
def metadatos():
    return leer_metadatos(METADATOS)

@st.cache_resource(show_spinner="Cargando modelo de clasificación...")
def obtener_modelo(ruta, ultima_modificacion, clases):
    return cargar_modelo(ruta, clases)


def llave_api():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError, AttributeError):
        return ""


def grafico_probabilidades(resultados):
    datos = pd.DataFrame([
        {"Condición": NOMBRES[c], "Probabilidad (%)": round(p * 100, 2)}
        for c, p in resultados
    ]).sort_values("Probabilidad (%)")
    fig = px.bar(datos, x="Probabilidad (%)", y="Condición", orientation="h",
                 color="Probabilidad (%)", color_continuous_scale=["#d5ede4", "#2e9d86", "#0d6359"],
                 text="Probabilidad (%)")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(height=345, showlegend=False, coloraxis_showscale=False,
                      margin=dict(l=8,r=60,t=12,b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis_range=[0,108],
                      xaxis_title=None, yaxis_title=None, font=dict(size=13))
    fig.update_xaxes(showgrid=True, gridcolor="#e4eeeb", ticksuffix="%")
    return fig

st.markdown("""<div class="hero"><small>ENERGÍA SOLAR · VISIÓN ARTIFICIAL · IA GENERATIVA</small>
<h1>☀️ SolarCheck AI</h1>
<p>Analiza fotografías de paneles solares, conoce el resultado del modelo y recibe recomendaciones preventivas elaboradas con inteligencia artificial.</p></div>""",unsafe_allow_html=True)

try:
    meta = metadatos()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"No se pudo leer modelo/clases.json: {exc}")
    st.stop()
clases = meta["clases"]
disponible = PESOS.exists() and PESOS.is_file()

with st.sidebar:
    st.markdown("## ☀️ SolarCheck AI")
    st.caption("Clasificador educativo de 6 condiciones en paneles fotovoltaicos")
    st.divider()
    st.markdown("**Modelo**")
    st.write("EfficientNet-B0 (timm) · fine-tuning")
    st.metric("Accuracy en TEST", f"{float(meta['test_accuracy'])*100:.1f}%")
    st.metric("Balanced accuracy en TEST", f"{float(meta['test_balanced_accuracy'])*100:.1f}%")
    st.caption("Métricas registradas en clases.json de la exportación más reciente de Colab; no son confianza de una imagen.")
    if disponible:
        st.success("Pesos locales detectados")
    else:
        st.warning("Falta el archivo modelo/modelo.pt")
    st.divider()
    st.caption("Proyecto académico · Se requiere revisión humana antes de cualquier intervención.")

tab_analizar, tab_modelo, tab_guia = st.tabs(["🔎 Analizar imagen", "📊 Resultados del modelo", "📘 Proyecto y seguridad"])

with tab_analizar:
    if not disponible:
        st.warning("**Análisis aún no disponible.** El entrenamiento ya está documentado, pero debes colocar `modelo/modelo.pt` exportado desde Colab. No se mostrarán predicciones inventadas.")
    st.subheader("1. Carga una fotografía")
    st.write("Fotografía JPG o PNG de un panel solar. Prueba con imágenes representativas de las seis clases.")
    archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if archivo is not None:
        contenido = archivo.getvalue()
        if len(contenido) > 10 * 1024 * 1024:
            st.error("La imagen supera los 10 MB. Prueba con otra de menor tamaño.")
            st.stop()
        try:
            imagen = ImageOps.exif_transpose(Image.open(BytesIO(contenido))).convert("RGB")
            imagen.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"No se pudo abrir la imagen: {exc}")
            st.stop()
        id_foto = hashlib.sha256(contenido).hexdigest()
        if st.session_state.get("foto_actual") != id_foto:
            st.session_state["foto_actual"] = id_foto
            st.session_state.pop("texto_recomendacion", None)
        izquierda, derecha = st.columns([0.9, 1.1], gap="large")
        with izquierda:
            st.image(imagen, caption="Imagen proporcionada por el usuario", use_container_width=True)
        with derecha:
            if disponible:
                try:
                    modelo = obtener_modelo(str(PESOS), PESOS.stat().st_mtime_ns, tuple(clases))
                    resultados = clasificar(imagen, modelo, clases)
                except Exception as exc:
                    st.error(f"El modelo no pudo cargar o inferir: {exc}")
                    st.stop()
                etiqueta, probabilidad = resultados[0]
                st.subheader("2. Resultado de clasificación")
                st.metric("Condición más probable", NOMBRES[etiqueta])
                st.metric("Confianza de esta imagen", f"{probabilidad*100:.1f}%")
                if probabilidad < UMBRAL_REVISION:
                    st.warning("**Confianza baja:** repetir fotografía y pedir revisión humana. No tomar decisiones solo con esta salida.")
                else:
                    st.info("La confianza no equivale a certeza: confirma el estado del panel con revisión humana.")
                st.plotly_chart(grafico_probabilidades(resultados), use_container_width=True, config={"displayModeBar":False})
            else:
                st.info("Aquí aparecerán la clase, la confianza y las probabilidades al incorporar el archivo de pesos.")
        if disponible:
            st.divider()
            st.subheader("3. Recomendación generada por IA")
            st.write("Gemini recibe únicamente la etiqueta y las probabilidades. **No recibe tu fotografía.**")
            clave = llave_api()
            if not clave:
                st.info("Configura `GEMINI_API_KEY` en los secretos de Streamlit para habilitar las recomendaciones.")
            if st.button("✨ Generar recomendación con Gemini", type="primary", disabled=(not bool(clave))):
                with st.spinner("Preparando recomendación de mantenimiento... "):
                    try:
                        st.session_state["texto_recomendacion"] = recomendacion_con_gemini(
                            clave, etiqueta, probabilidad, resultados
                        )
                    except Exception as exc:
                        st.error("No fue posible consultar Gemini. Revisa la clave, cuotas y conexión. " + str(exc))
            if st.session_state.get("texto_recomendacion"):
                st.success("Recomendación generada")
                st.markdown(st.session_state["texto_recomendacion"])
                st.caption("Texto generado automáticamente; no constituye inspección, diagnóstico ni instrucción de reparación.")
    else:
        st.info("Sube una imagen para iniciar el análisis visual.")

with tab_modelo:
    st.subheader("Cómo se evaluó el clasificador")
    st.write("Se experimentó con **CLIP sin entrenamiento adicional** y **EfficientNet-B0 ajustado**. El entrenamiento del último modelo se ejecutó nuevamente en Colab; su archivo de metadatos registra métricas actualizadas. Se conserva también la evaluación anterior para trazabilidad.")
    col1,col2,col3 = st.columns(3)
    col1.metric("CLIP · accuracy (sesión previa)", "48,1%")
    col2.metric("EfficientNet-B0 · accuracy (exportación reciente)", f"{float(meta['test_accuracy'])*100:.1f}%")
    col3.metric("Balanced accuracy · exportación reciente", f"{float(meta['test_balanced_accuracy'])*100:.1f}%")
    st.caption("CLIP proviene del notebook anterior; EfficientNet proviene del JSON exportado desde Colab en una nueva ejecución. Antes de afirmar una comparación final de la misma sesión, actualiza también el notebook ejecutado con sus nuevas salidas.")
    with st.expander("Ver análisis detallado de una ejecución ANTERIOR (accuracy 74,0%)", expanded=False):
        st.info("Estos gráficos son del notebook anterior: NO describen necesariamente los pesos recién exportados (accuracy registrada: 87,8%).")
        disponibles = [ROOT/"resultados"/"comparacion_recall.png", ROOT/"resultados"/"efficientnet_matriz.png"]
        grafs = st.columns(2)
        for cont, ruta in zip(grafs, disponibles):
            if ruta.exists():
                with cont:
                    st.image(str(ruta), caption=("Recall por clase · sesión anterior" if "recall" in ruta.name else "Matriz de confusión · sesión anterior"), use_container_width=True)
        st.markdown("En esa ejecución anterior, EfficientNet confundió **5 de 29** paneles con polvo con paneles limpios (recall Dusty 48,3%).")
    st.markdown("**Riesgo principal:** no detectar daño eléctrico o físico puede retrasar una revisión técnica; ninguna clase predicha constituye una certificación de seguridad.")
    st.warning("Limitaciones: dataset de una única fuente, clases desbalanceadas (69-193 imágenes) y licencia indicada como desconocida. El modelo no está validado en escenarios industriales ni reemplaza inspecciones técnicas.")

with tab_guia:
    st.subheader("Propósito y condiciones")
    st.write("SolarCheck AI es un prototipo académico para apoyar la priorización de inspecciones en paneles fotovoltaicos. Reconoce seis estados: limpio, polvo, excrementos de aves, nieve, daño físico y posible daño eléctrico.")
    st.markdown("**Flujo:** imagen → EfficientNet-B0 → probabilidades → solicitud voluntaria → Gemini → recomendación cautelosa.")
    st.warning("**Solo con fines educativos.** El sistema puede equivocarse. Nunca manipules instalaciones energizadas ni realices trabajos en altura por una sugerencia de esta aplicación. Solicita evaluación a profesionales autorizados.")
    st.caption("Datos: https://www.kaggle.com/datasets/pythonafroz/solar-panel-images · Origen declarado en el notebook original: Afroz (pythonafroz). Licencia no especificada al momento del registro académico.")
