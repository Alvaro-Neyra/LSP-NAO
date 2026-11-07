# Especificación: 001 - Recolección Colaborativa de Datos LSP

## QUÉ: Requerimientos Funcionales
1. **Captura bimanual en tiempo real:** Detectar hasta 2 manos simultáneas usando MediaPipe Hands sobre video a 25 cuadros por segundo.
2. **Personalización por integrante:** Cada alumno debe almacenar su captura en `data/raw/dataset_lsp_{nombre}.csv`.
3. **Tope estricto de muestras:** Pausar automáticamente al alcanzar exactamente 300 muestras por clase ($2,400$ muestras totales por persona).
4. **Respaldo defensivo:** Evitar sobreescritura accidental y proteger datos anteriores.
