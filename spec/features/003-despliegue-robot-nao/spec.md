# Especificación: 003 - Integración y Despliegue en el Robot NAO

## QUÉ: Requerimientos Funcionales
1. **Transmisión de Video en Tiempo Real:** Adquirir cuadros desde la cámara de la cabeza del robot mediante `ALVideoDevice` o desde cámara web para inferencia.
2. **Inferencia Continua:** Extraer landmarks con MediaPipe, inferir con `modelo_lsp_ganador.joblib` y filtrar por umbral de confianza ($>85\%$).
3. **Respuesta Robótica Multimodal:**
   * **Voz:** El robot debe pronunciar la traducción de la seña en español usando `ALTextToSpeech`.
   * **Movimiento:** Realizar gestos articulares de confirmación o adoptar posturas con `ALMotion`.
4. **Manejo de Cooldown:** Filtro temporal de 3 segundos para evitar repeticiones excesivas de síntesis de voz sobre la misma seña.
