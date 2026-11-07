# ADR 0001: Representación Bimanual Fija de 126 Dimensiones con Zero-Padding

## Contexto
El reconocimiento de Lengua de Señas Peruana (LSP) abarca tanto gestos unimanuales (`YO`, `TU`, `AGUA`) como gestos bimanuales (`AYUDA`, `NECESITAR`, `QUERER`). En la implementación inicial con MediaPipe Hands, el vector de características era dinámico y se sobreescribía con una sola mano (63 columnas), lo que impedía clasificar señas bimanuales.

## Decisión
Adoptar una matriz de diseño de **longitud fija con 126 columnas numéricas**:
* 63 valores para la mano izquierda (`lh_lm_...`).
* 63 valores para la mano derecha (`rh_lm_...`).
* Cada mano se normaliza independientemente respecto a su propia muñeca y se escala invariantemente.
* Si un gesto es unimanual, la mitad correspondiente a la mano ausente se completa con ceros (`0.0`).

## Consecuencias
* **Positivas:** Permite que los algoritmos clásicos de Machine Learning (SVM, Random Forest) aprendan de un dataset tabular rectangular y consistente. El modelo discrimina automáticamente si una seña requiere una mano o dos manos en función de la presencia de ceros.
* **Negativas:** Se añade una leve carga computacional al detector de MediaPipe al evaluar dos manos simultáneas, pero se mantiene holgadamente dentro de los márgenes de tiempo real (>25 FPS).
