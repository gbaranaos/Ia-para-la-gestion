# Ficha de datos · SolarCheck AI

**Nombre:** Solar Panel Images - Clean and Faulty Images.

**Autor atribuido en el notebook original:** Afroz (`pythonafroz`).

**Fuente:** https://www.kaggle.com/datasets/pythonafroz/solar-panel-images

**Obtención:** descarga desde Kaggle mediante `kagglehub`; no se proporcionó descripción verificable de la forma en que se capturaron originalmente las fotografías.

**Licencia:** el notebook aportado la registra como "Unknown / no especificada". Verificar las condiciones antes de redistribuir fotos o subir ejemplos a un repositorio público.

**Clases:** Bird-drop (191), Clean (193), Dusty (190), Electrical-damage (103), Physical-Damage (69), Snow-Covered (123): total 869. La clase adicional New (16) se descartó por baja representación.

**División:** train 608, val 130, test 131; estratificación con `train_test_split`, `random_state=42`. Ver detalle por clase en `division_datos.csv`.

**Desbalance:** relación 193/69 ≈ 2,8 entre la clase más grande y la menor; se emplea CrossEntropyLoss ponderada durante el ajuste.

**Limitaciones:** una fuente, faltan metadatos de captura y de validación industrial; 10 imágenes de daño físico en test no bastan para inferencias de seguridad; no está descartada similitud entre imágenes; permiso de distribución no aclarado.

**Uso responsable:** los resultados no son diagnósticos ni sustituyen revisión humana o de personal técnico acreditado.
