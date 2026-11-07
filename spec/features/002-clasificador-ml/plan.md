# Plan Técnico: 002 - Entrenamiento y Comparativa de Modelos ML

## CÓMO: Enfoque Técnico
* **Prevención de Data Leakage:** Encapsular `StandardScaler` y `SVC` dentro de un `Pipeline` de Scikit-Learn para que el ajuste de media y varianza se compute estrictamente sobre el subconjunto de entrenamiento.
* **Cronometraje de Latencia:** Evaluar el estimador con 300 muestras individuales registrando el tiempo con `time.perf_counter()` en milisegundos.
* **Empaquetado de Producción:** Guardar en `models/modelo_lsp_ganador.joblib` un diccionario que contenga el estimador, mapa de clases, nombres de columnas y métricas registradas.
