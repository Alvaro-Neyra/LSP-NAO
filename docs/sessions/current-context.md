# Contexto Actual y Memoria Viva del Proyecto (Current Context)
Ultima actualizacion: 2026-11-07

## 1. Estado del Proyecto
* **Fase Actual:** Integracion Robótica y Banco de Pruebas de Inferencia completados.
* **Modelo Activo:** `models/modelo_lsp_ganador.joblib` (SVM RBF, 126 dimensiones, 96.02% Accuracy, 0.408 ms latencia).
* **Clases:** `['YO', 'TU', 'NECESITAR', 'QUERER', 'AGUA', 'AYUDA', 'MEDICINA', 'ENVIAR']`.
* **Pruebas:** 9 pruebas unitarias automatizadas pasando con éxito (`uv run pytest`).

## 2. Componentes Clave en Funcionamiento
* **Banco de Pruebas Autónomo:** `src/vision/probar_modelo_en_vivo.py`
  - Captura automática por estabilidad (0.75 s, >= 85%).
  - Mosaico y galería final con Matplotlib al presionar Q.
* **Traductor Integrado con Robot NAO:**
  - `src/robot/bridge_nao_server.py` (Python 2.7 / PyNAOqi / Choregraphe puerto 54180).
  - `src/robot/app_inferencia_nao.py` (Python 3.12 / MediaPipe / Socket TCP 50005).
  - Orquestador: `uv run python main.py --nao` (o menú opción 6).

## 3. Restricciones Críticas Recordadas
* Código Python sin la letra enie (reemplazada por 'n' en variables, comentarios y textos de voz).
* Entorno dual desacoplado: Python 3.12 (inferencia) <-> Socket TCP <-> Python 2.7 (NAOqi).
