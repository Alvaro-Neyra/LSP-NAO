# Plan Técnico: 001 - Recolección Colaborativa de Datos LSP

## CÓMO: Enfoque de Implementación
* **Extracción geométrica:** Utilizar `mp.solutions.hands.Hands(max_num_hands=2)`.
* **Separación de lateralidad:** Evaluar `results.multi_handedness` para canalizar los puntos hacia el slot izquierdo (`lh_`) o derecho (`rh_`).
* **Sincronización:** Leer el conteo de muestras preexistentes en disco para permitir sesiones de captura pausables y reanudables.
