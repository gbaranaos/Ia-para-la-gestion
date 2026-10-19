"""Inferencia reproducible de EfficientNet-B0; mismo preprocesado de Colab."""
from pathlib import Path
import json

NOMBRES = {
    "Bird-drop": "Excrementos de aves",
    "Clean": "Panel limpio",
    "Dusty": "Acumulación de polvo",
    "Electrical-damage": "Posible daño eléctrico",
    "Physical-Damage": "Daño físico",
    "Snow-Covered": "Cobertura de nieve",
}


def leer_metadatos(ruta):
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    clases = datos.get("clases")
    if not isinstance(clases, list) or sorted(clases) != sorted(NOMBRES):
        raise ValueError("Las seis clases no coinciden con el modelo entrenado.")
    if len(set(clases)) != len(clases):
        raise ValueError("Hay clases duplicadas.")
    return datos


def cargar_modelo(ruta_pesos, clases):
    """Solo requiere pesos locales; nunca descarga ni reemplaza por pesos aleatorios."""
    import torch
    import timm
    ruta_pesos = Path(ruta_pesos)
    if not ruta_pesos.is_file():
        raise FileNotFoundError("Falta modelo/modelo.pt: expórtalo desde Colab.")
    modelo = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(clases))
    estado = torch.load(ruta_pesos, map_location="cpu", weights_only=True)
    modelo.load_state_dict(estado, strict=True)
    modelo.eval()
    return modelo


def clasificar(imagen, modelo, clases):
    import torch
    from torchvision import transforms
    from PIL import ImageOps
    normalizacion = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    preprocesado = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), normalizacion
    ])
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")
    tensor = preprocesado(imagen).unsqueeze(0)
    with torch.inference_mode():
        logits = modelo(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    if len(probs) != len(clases):
        raise ValueError("La cantidad de salidas no coincide con las clases.")
    return sorted(zip(clases, probs), key=lambda par: par[1], reverse=True)


def prompt_responsable(clase, confianza, probabilidades):
    if clase not in NOMBRES:
        raise ValueError("Clase no reconocida.")
    otras = ", ".join(f"{NOMBRES[c]}: {p:.1%}" for c, p in probabilidades[:4])
    baja = confianza < 0.65
    aviso = (
        "La confianza es baja: prioriza una nueva foto y revisión técnica; no concluyas la causa."
        if baja else
        "La probabilidad no garantiza que la etiqueta sea correcta; sugiere revisión humana."
    )
    return f"""Resultado de un clasificador de imágenes de paneles solares (NO inspección técnica):
Clase estimada: {NOMBRES[clase]} ({clase}).
Confianza del modelo para esta imagen: {confianza:.1%} (no equivale a accuracy de prueba).
Probabilidades más altas: {otras}.
{aviso}
Escribe en español sencillo una recomendación de mantenimiento preventiva y contextual de 90 a 140 palabras.
Incluye: 1) interpretación cautelosa, 2) recomendación inmediata segura,
3) cuándo escalar a personal técnico, 4) recordatorio de uso educativo.
Nunca asegures un diagnóstico, jamás des instrucciones de manipulación de equipos energizados,
ni indiques subir a un techo o realizar limpieza con riesgo de caída.
Para posible daño eléctrico o físico prioriza inspección por personal calificado;
para limpieza prioriza seguir protocolos de seguridad.
No inventes medidas, causas ni observaciones que el modelo no ha proporcionado."""


def recomendacion_con_gemini(api_key, clase, confianza, probabilidades):
    """Envía SOLO resultados textuales, no fotografías, a Gemini."""
    from google import genai
    from google.genai import types
    prompt = prompt_responsable(clase, confianza, probabilidades)
    cliente = genai.Client(api_key=api_key)
    try:
        resp = cliente.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.25,
                max_output_tokens=650,
                system_instruction=(
                    "Eres un asistente educativo de mantenimiento fotovoltaico. "
                    "No has visto la fotografía. Solo conoces probabilidades de un modelo. "
                    "Debes ser prudente con seguridad eléctrica y trabajos en altura. "
                    "No proporciones pasos técnicos de reparación ni ignores la revisión humana."
                ),
            ),
        )
        if not resp.text or not resp.text.strip():
            raise RuntimeError("Gemini no devolvió texto; revisa la configuración de la API.")
        return resp.text.strip()
    finally:
        cliente.close()
