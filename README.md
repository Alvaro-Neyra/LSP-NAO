# Inteligencia Artificial (UPC 2026-2)

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/fc/UPC_logo_transparente.png" alt="Logo UPC" width="220"/>
</p>

## Práctica N.° 2 - Implementación de un Modelo de Machine Learning en el Robot NAO
### **Tema: Reconocimiento en Tiempo Real de Lengua de Señas Peruana (LSP)**

---

## 1. Descripción del Proyecto

El presente proyecto consiste en el diseño, desarrollo, entrenamiento y despliegue de un sistema de visión artificial y Machine Learning para el reconocimiento de **Lengua de Señas Peruana (LSP)** en tiempo real, integrado con el robot humanoide **NAO**.

El sistema procesa secuencias de video mediante **MediaPipe Hands** para extraer las coordenadas tridimensionales normalizadas de las manos (126 características por muestra, contemplando gestos unimanuales y bimanuales con técnica de *zero-padding*). La clasificación se realiza mediante un modelo de aprendizaje supervisado optimizado que comanda al robot NAO para que actúe como un intérprete accesible, verbalizando las palabras traducidas mediante síntesis de voz (`ALTextToSpeech`) y ejecutando acciones motrices de confirmación (`ALMotion`).

---

## 2. Integrantes del Grupo

| Código | Integrante | Rol en el Proyecto | Responsabilidad Principal |
| :---: | :--- | :--- | :--- |
| **U202211741** | **Alvaro Alberto Neyra Salazar** | **Machine Learning & Architecture Lead** | Diseño de la arquitectura de datos, pipelines de entrenamiento en Scikit-Learn y orquestación general. |
| **U202412310** | **Victor Piero Arapa Titi** | **Data Collection & Preprocessing** | Adquisición de muestras con MediaPipe, control de balanceo y preprocesamiento de coordenadas. |
| **U202413112** | **Salvador Arturo Diaz Miñano** | **Evaluation & Validation Analyst** | Análisis empírico de métricas (Accuracy, F1-Score, latencia) y diseño de la suite de pruebas automatizadas. |
| **U20241e410** | **Rai Jeferson Manrique Anaya** | **Robot Integration Engineer** | Integración del runtime de inferencia (Python 3) con la arquitectura de control de NAOqi (Python 2.7). |

---

## 3. Conjunto de Datos (Dataset LSP Bimanual)

El corpus fue construido colaborativamente por los 5 integrantes del equipo, acumulando un total de **12,046 muestras** distribuidas en 8 clases esenciales de comunicación:

* **Características ($X$):** 126 columnas numéricas continuas correspondientes a las coordenadas relativas y normalizadas $[x, y, z]$ de los 21 landmarks de la mano izquierda (`lh_lm_0_x` a `lh_lm_20_z`) y derecha (`rh_lm_0_x` a `rh_lm_20_z`).
* **Etiquetas ($y$):** 8 categorías balanceadas:
  1. `YO` (1,546 muestras)
  2. `TU` (1,500 muestras)
  3. `NECESITAR` (1,500 muestras)
  4. `QUERER` (1,500 muestras)
  5. `AGUA` (1,500 muestras)
  6. `AYUDA` (1,500 muestras)
  7. `MEDICINA` (1,500 muestras)
  8. `ENVIAR` (1,500 muestras)
* **Partición de Datos:** 80% para entrenamiento (9,636 muestras) y 20% para evaluación no vista (2,410 muestras), utilizando una partición estratificada con semilla fija (`random_state=42`).

---

## 4. Comparativa de Modelos y Resultados de Entrenamiento

Se evaluaron experimentalmente dos familias de algoritmos supervisados con el objetivo de balancear precisión de clasificación y latencia de respuesta para robótica interactiva:

1. **Support Vector Machine (SVC):** Implementado mediante un `Pipeline` con `StandardScaler` y función de base radial (RBF) con $C=10.0$ y $\gamma=\text{'scale'}$.
2. **Random Forest Classifier:** Ensamble de 100 árboles de decisión con criterio Gini y paralelismo activo.

### Resultados Generales

| Métrica / Parámetro | Support Vector Machine (RBF) [Ganador] | Random Forest Classifier |
| :--- | :---: | :---: |
| **Exactitud Global (Accuracy)** | **96.02%** | 94.94% |
| **F1-Score Macro** | **96.03%** | 94.94% |
| **F1-Score Weighted** | **96.03%** | 94.94% |
| **Latencia de Inferencia por Muestra** | **0.408 ms** | 15.248 ms |
| **Tamaño del Modelo Serializado** | **3.07 MB** | 17.85 MB |
| **Tiempo de Entrenamiento** | 4.42 s | 0.46 s |

### Desglose de F1-Score por Clase de Seña (Conjunto de Evaluación)

| Clase (LSP) | F1-Score SVM (RBF) | F1-Score Random Forest | Soporte (Muestras Test) |
| :--- | :---: | :---: | :---: |
| **YO** | **95.56%** | 94.94% | 310 |
| **TU** | **93.67%** | 91.69% | 300 |
| **NECESITAR** | **93.88%** | 92.80% | 300 |
| **QUERER** | **98.32%** | 97.14% | 300 |
| **AGUA** | **96.82%** | 93.29% | 300 |
| **AYUDA** | 96.82% | **97.49%** | 300 |
| **MEDICINA** | **98.49%** | 98.00% | 300 |
| **ENVIAR** | **94.67%** | 94.18% | 300 |

### Criterio de Selección del Modelo de Producción
**Support Vector Machine (RBF)** fue seleccionado como el modelo ganador definitivo para su despliegue en el robot NAO debido a:
* **Rendimiento:** Mayor F1-Score Macro (96.03% frente a 94.94%).
* **Latencia Ultra-Baja:** La inferencia toma solo **0.408 milisegundos por muestra**, resultando **~37 veces más rápido** que Random Forest, lo cual permite procesar video a más de 60 FPS sin degradar la respuesta motriz.
* **Huella de Memoria Reducida:** Requiere solo **3.07 MB** frente a los 17.85 MB del ensamble de árboles.

---

## 5. Ubicación de Archivos y Resultados en el Repositorio

Todos los artefactos, datos y reportes generados se encuentran organizados en la siguiente estructura:

* **Métricas y Resultados Tabulados:** `reports/metricas_comparativas.txt`
* **Gráficas Editoriales (300 DPI):**
  * Matriz de Confusión Normalizada: `reports/figures/matrices_confusion_lsp.png`
  * Comparativa de F1 por Clase: `reports/figures/comparativa_f1_por_clase.png`
* **Modelo Serializado para Despliegue:** `models/modelo_lsp_ganador.joblib`
* **Datasets:**
  * Archivos individuales de integrantes: `data/raw/`
  * Dataset consolidado y balanceado: `data/processed/dataset_lsp_consolidado.csv`
* **Cuaderno Interactivo en la Nube:** `notebooks/entrenamiento_lsp_colab.ipynb`
* **Código Fuente Modular:**
  * Captura de señas: `src/vision/capturar_dataset_lsp.py`
  * Banco de pruebas en vivo y captura de evidencias: `src/vision/probar_modelo_en_vivo.py`
  * Pipeline de datos y ML: `src/pipeline/`
  * Control y puente con el robot NAO: `src/robot/`
    * Servidor socket y control de proxies NAOqi (Python 2.7): `src/robot/bridge_nao_server.py`
    * Inferencia, máquina de estados y cliente socket (Python 3): `src/robot/app_inferencia_nao.py`
    * Script de pruebas exploratorias de articulaciones: `src/robot/codlines_nao.py`
  * Lanzadores directos en raíz: `bridge_nao_server.py` y `app_inferencia_nao.py` (atajos de ejecución rápida hacia `src/robot/`)
* **Cuadernos Interactivos:**
  * Entrenamiento comparativo en Colab: `notebooks/entrenamiento_lsp_colab.ipynb`
  * Banco de pruebas en vivo con cámara WebRTC: `notebooks/banco_pruebas_lsp_colab.ipynb`
* **Pruebas Automatizadas:** `tests/` (`test_data_integrity.py`, `test_model_inference.py`, `test_nao_integration.py`)

---

## 6. Instrucciones de Ejecución

El proyecto cuenta con un orquestador central en la raíz (`main.py`):

```powershell
# Ejecución interactiva (Menú en consola)
uv run python main.py

# Ejecución por comandos directos
uv run python main.py --capturar   # Iniciar captura con MediaPipe
uv run python main.py --unificar   # Unificar datasets individuales
uv run python main.py --entrenar   # Entrenar y evaluar modelos
uv run python main.py --probar     # Probar modelo en vivo con captura de evidencias
uv run python main.py --nao        # Iniciar traductor integrado con Robot NAO
uv run python main.py --test       # Correr suite de pruebas con pytest
```

### Ejecución de la Integración con Robot NAO (Dos Terminales)

Debido a que el SDK de NAOqi opera en Python 2.7 (32-bit) y la inferencia en Python 3:

```powershell
# Terminal 1 (Python 2.7): Controlador del Robot (físico o virtual en Choregraphe)
python bridge_nao_server.py

# Terminal 2 (Python 3): Inferencia y cámara web
uv run python app_inferencia_nao.py
```

---

## 7. Licencia

Este proyecto se distribuye bajo la **Licencia MIT** con fines estrictamente académicos para la Universidad Peruana de Ciencias Aplicadas (UPC).
