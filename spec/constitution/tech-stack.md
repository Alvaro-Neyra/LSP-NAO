# Pila Tecnológica y Convenciones (Tech-Stack)

## Entornos de Ejecución

### 1. Entorno de Machine Learning y Visión (Python 3.12)
* **Gestor de paquetes y entornos:** `uv` (v0.11+).
* **Entorno virtual:** `.venv` (CPython 3.12.3 de 64 bits).
* **Librerías principales:**
  * `mediapipe == 0.10.14`: Extracción de landmarks de manos bimanuales (solución clásica estable `mp.solutions.hands`).
  * `opencv-python == 4.11.0.86`: Adquisición de video, renderizado HUD y manipulación de cuadros.
  * `scikit-learn == 1.9.1`: Algoritmos de clasificación (`SVC`, `RandomForestClassifier`), pipelines y métricas.
  * `numpy == 1.26.4`: Manipulación de tensores y vectores algebraicos.
  * `pandas == 3.0.6`: Ingesta, balanceo y exportación tabular de datasets.
  * `joblib == 1.6.0`: Serialización binaria de pipelines y metadatos.
  * `matplotlib == 3.11.2` y `seaborn == 0.13.2`: Visualización editorial a 300 DPI.

### 2. Entorno del Robot NAO (Python 2.7)
* **Intérprete:** Python 2.7.x (32 bits, ruta estándar `C:\Python27\python.exe`).
* **SDK:** `pynaoqi == 2.8.x`.
* **Módulos NAOqi utilizados:**
  * `ALTextToSpeech`: Síntesis de voz en español para verbalizar la traducción del gesto.
  * `ALMotion` y `ALRobotPosture`: Control cinemático de articulaciones y posturas predefinidas (`StandInit`).
  * `ALVideoDevice`: Transmisión remota de cuadros desde la cámara de la cabeza del robot.
