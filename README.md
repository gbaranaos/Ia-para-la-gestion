# ☀️ SolarCheck AI

**Clasificación visual educativa de paneles solares con EfficientNet-B0 y recomendaciones con Gemini.**

> **Estado actualizado (19/10/2026):** el archivo real `modelo/modelo.pt` ya fue integrado desde `solarcheck.zip` del grupo. La exportación reciente registra accuracy y balanced accuracy de **87,8%**. El notebook y las matrices originales muestran resultados de una ejecución previa (**74,0%**), por lo que deben actualizarse con las salidas de la nueva ejecución antes de presentarlos como evaluación final del modelo. La publicación y la clave de Gemini siguen pendientes.

## 1. Problema

Una inspección visual manual de paneles solares puede tardar y variar entre evaluadores. SolarCheck AI propone una clasificación preliminar de fotografías para orientar la priorización de revisión, limpieza y mantenimiento. **No certifica ausencia de daño ni sustituye el criterio técnico.**

**Flujo:** 📷 Imagen → 🧠 EfficientNet-B0 fine-tuned → 📊 Clase + probabilidades → 💬 Gemini (solo texto de probabilidades) → ✅ Recomendación preventiva cautelosa.

## 2. Datos

- **Origen:** Kaggle, [Solar Panel Images - Clean and Faulty Images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images), autor señalado en el notebook: Afroz (`pythonafroz`).
- **Adquisición:** `kagglehub.dataset_download("pythonafroz/solar-panel-images")`. La metodología de captura original no está documentada en el material recibido.
- **Licencia:** declarada como **Unknown / no especificada** en el notebook; comprobar los términos actuales antes de redistribuir imágenes.
- **Se descartó `New`**, que contenía 16 imágenes. Quedaron **869 imágenes en seis clases**.

| Clase | Entrenamiento | Validación | Prueba | Total |
|---|---:|---:|---:|---:|
| Bird-drop | 134 | 28 | 29 | 191 |
| Clean | 135 | 29 | 29 | 193 |
| Dusty | 133 | 28 | 29 | 190 |
| Electrical-damage | 72 | 16 | 15 | 103 |
| Physical-Damage | 48 | 11 | 10 | 69 |
| Snow-Covered | 86 | 18 | 19 | 123 |
| **Total** | **608** | **130** | **131** | **869** |

**Separación estratificada:** 70% entrenamiento, aproximadamente 15% validación y 15% prueba, mediante dos llamadas a `train_test_split` con `stratify` y `random_state=42`. El orden del directorio, librerías y GPU pueden influir en una nueva corrida. Las imágenes se transforman a RGB y tamaño 224x224 para la inferencia.

**Sesgos y limitaciones:** una única fuente de datos; desequilibrio entre 69 y 193 imágenes por clase; solo 10 fotografías de daño físico en la muestra de prueba; condiciones de captura no controladas; no se investigó posible correlación entre fotografías similares. No se dispone de validación de campo ni derechos para redistribuir libremente el dataset.

## 3. Resultados y trazabilidad de las ejecuciones

**Resultados de la exportación reciente del modelo que utiliza la app:**
- Accuracy en prueba: **87,8%** (registrado en `solarcheck.zip/modelo/clases.json`).
- Balanced accuracy en prueba: **87,8%** (misma fuente).
- Los pesos se copiaron sin cambios desde el ZIP: archivo `modelo/modelo.pt` (SHA-256 documentado en `modelo/clases.json`).

**Resultados de la ejecución anterior (notebook y gráficos incluidos como evidencia histórica):** CLIP 48,1%, EfficientNet-B0 74,0%, balanced accuracy 76,8% y recall Dusty 48,3%. Los gráficos de confusión de esa ejecución **no deben atribuirse a los pesos actuales**. La prueba de la nueva ejecución debe guardarse en el notebook al volver a ejecutarlo.

## 3.1 Comparación de modelos

- **Línea base:** CLIP `openai/clip-vit-base-patch32`, comparación zero-shot de descripciones en inglés contra las 131 imágenes de prueba.
- **Modelo elegido:** `timm.efficientnet_b0` preentrenado con ImageNet y última capa reemplazada por 6 salidas; fine-tuning con AdamW, 15 épocas, class weights y data augmentation únicamente en entrenamiento; selección del mejor estado mediante `balanced_accuracy` en validación.
- **Criterio:** mismas imágenes de prueba y mismas métricas para ambos enfoques.

| Métrica registrada en el notebook original (TEST) | CLIP | EfficientNet-B0 |
|---|---:|---:|
| Accuracy | 48,1% | **74,0% (ejecución anterior)** |
| Balanced accuracy | 51,4% | **76,8% (ejecución anterior)** |
| Recall de Dusty | 3,4% | **48,3%** |
| Dusty clasificados como Clean | 14 de 29 | **5 de 29** |

**Interpretación de la ejecución anterior:** EfficientNet mejora **25,9 puntos porcentuales** en accuracy frente a CLIP, aunque aún deja pasar un número relevante de casos con polvo. Si un panel con daño eléctrico se interpreta como una situación rutinaria, el costo en seguridad podría ser elevado; no se dispone de datos para cuantificar un costo monetario. Las probabilidades de softmax **no están calibradas**, por lo que no son una garantía de diagnóstico. El conjunto de prueba debe reservarse para la evaluación final y no para escoger parámetros iterativamente.

Las imágenes de matrices de confusión y recalls de la ejecución anterior están en `resultados/` y no representan necesariamente el modelo exportado más recientemente.

## 4. Archivos del repositorio

```text
SolarCheck_AI/
├── app.py                        # Interfaz Streamlit
├── inferencia.py                 # Modelo, clasificación y Gemini
├── integrar_exportacion.py      # Copiar pesos de solarcheck.zip de Colab
├── verificar_proyecto.py        # Prueba de carga de modelo
├── requirements.txt
├── README.md
├── GUIA_PUBLICACION.md
├── GUION_EXPOSICION.md
├── notebooks/entrenamiento.ipynb
├── modelo/clases.json           # Etiquetas y métricas registradas
├── modelo/modelo.pt             # INCLUIDO: pesos reales del ZIP recibido
├── resultados/                  # Métricas, gráficos y recuentos reales
├── ejemplos/                    # Opcional: imágenes autorizadas
├── tests/test_inferencia.py
├── presentacion/                # PDF y PowerPoint editables
└── .streamlit/config.toml
```

Se incluye `modelo.pt` real, incorporado desde el ZIP del grupo. No se incluyen fotografías originales, por la licencia no aclarada. **No redistribuir imágenes sin autorización.**

## 5. Ejecutar el entrenamiento en Google Colab

1. Abre `notebooks/entrenamiento.ipynb` en [Google Colab](https://colab.research.google.com/) y habilita GPU si es posible.
2. Ejecuta las celdas en orden. `kagglehub` descargará el dataset (se requiere conexión).
3. Al finalizar, la celda de exportación descargará **`solarcheck.zip`**: contiene `modelo/modelo.pt`, `modelo/clases.json`, gráficos y hasta 6 ejemplos. La integración importa **solo los pesos** para no publicar imágenes cuya licencia no está aclarada.
4. Descomprime o usa el comando desde el directorio del proyecto:

```bash
python integrar_exportacion.py /ruta/de/solarcheck.zip
```

5. Verifica el peso real:

```bash
python verificar_proyecto.py
```

Si ya tienes el `solarcheck.zip` del entrenamiento anterior, **no necesitas reentrenar**: úsalo directamente. Importa únicamente los pesos y confirma que las seis clases están en el orden esperado.

## 6. Probar la aplicación en tu computador

Se recomienda Python 3.11 o 3.12:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

La URL local habitual es `http://localhost:8501`. Selecciona una imagen JPG/PNG. Se muestra una clase predicha, su probabilidad *softmax*, el gráfico completo, y un botón para Gemini. **Con pesos ausentes**, se muestra un aviso claro y no se simula una predicción.

### Activar Gemini

En [Google AI Studio](https://aistudio.google.com/apikey), genera una clave de API (el nivel gratuito depende de condiciones de Google). En Streamlit Cloud, abre la app → *Settings / Advanced settings → Secrets* y pega:

```toml
GEMINI_API_KEY = "AQUI_VA_TU_CLAVE_REAL"
```

En local, utiliza `.streamlit/secrets.toml` con ese mismo contenido (el archivo está bloqueado por `.gitignore`). También puedes configurar la variable de entorno `GEMINI_API_KEY`. **Nunca la subas a GitHub.**

Se usa `google-genai` con `gemini-2.5-flash`; Google puede cambiar disponibilidad/cuotas. El LLM recibe **solo etiqueta, confianza y probabilidades**, nunca la imagen. Con confianza inferior al 65%, el prompt solicita repetir la fotografía y revisión humana. No hace reparaciones ni instrucciones peligrosas.

## 7. Publicar gratis: GitHub y Streamlit Community Cloud

1. Crea un repositorio público de GitHub llamado `solarcheck-ai` y sube los archivos **incluyendo el archivo de pesos real** (debe pesar menos de 100 MB), pero **sin el dataset completo ni secretos**.
2. En [Streamlit Community Cloud](https://share.streamlit.io/), conecta GitHub → `Create app`, selecciona repositorio, rama principal y archivo `app.py`.
3. En configuración avanzada, usa Python 3.11/3.12 y carga `GEMINI_API_KEY` como secreto.
4. Despliega, prueba una imagen, el botón de Gemini y el enlace público desde otro dispositivo.
5. Entrega al profesor enlace a datos, repositorio, app y PDF en `presentacion/`.

**Importante:** este ZIP ya incluye los pesos, pero NO está publicado y no incluye la clave del usuario. Se requiere acción desde tus cuentas de GitHub, Streamlit y Google AI Studio.

## 8. Pruebas y controles

```bash
python -m unittest discover -s tests
python verificar_proyecto.py
```

Los test unitarios comprueban consistencia de clases y políticas básicas del prompt. El chequeo con pesos carga un modelo de verdad y hace una inferencia sintética **solo técnica**, no es una métrica de calidad. Para certamen, probar además fotografías que no hayan guiado el entrenamiento y verificar funcionamiento real del LLM.

## 9. Ética, privacidad y riesgos

Prototipo académico: no usar para declarar una instalación segura o para ordenar reparaciones de alto riesgo. La confianza del modelo por fotografía **no es** el accuracy global registrado de 87,8%. Toda sospecha de daño físico o eléctrico debe evaluarla personal calificado; no indicar que personas no capacitadas suban a techos o intervengan instalaciones energizadas. La clase que aparece no siempre corresponde a la condición real. La licencia del dataset es desconocida; la app no transmite la fotografía a Gemini.

## 10. Declaración del uso de IA

Se utilizó **ChatGPT (OpenAI)** como apoyo para **revisar el notebook, escribir y probar código de la app, documentar, estructurar el repositorio, diseñar la presentación y preparar las recomendaciones responsables**. Los resultados antiguos y matrices de confusión proceden de las salidas del notebook de entrenamiento aportado por el grupo; la exportación reciente añade dos métricas nuevas registradas por Colab; ChatGPT **no reentrenó** el modelo ni generó pesos para él; integró el archivo exportado por el grupo. El equipo debe comprender, revisar y ser capaz de explicar todo el código; antes de entregar, registrar aquí los nombres de los integrantes y cualquier otra herramienta de IA utilizada.

**Integrantes:** [Completar nombres reales del equipo].

**Materia:** IA Generativa para la Gestión Industrial · Universidad del Desarrollo · 2026.
