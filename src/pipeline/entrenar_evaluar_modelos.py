"""Módulo de entrenamiento, evaluación comparativa y serialización de modelos de Machine Learning

para el reconocimiento de Lengua de Senas Peruana (LSP) aplicado a robótica con el Robot NAO.
"""

from dataclasses import dataclass
import io
import os
import time
from typing import Any
import joblib
import matplotlib
matplotlib.use("Agg")  # Permite renderizar y guardar gráficos sin requerir entorno de ventana interactivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ==============================================================================
# Constantes de Configuración del Experimento
# ==============================================================================
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ARCHIVO_DATASET_CONSOLIDADO = os.path.join(DIRECTORIO_RAIZ, "data", "processed", "dataset_lsp_consolidado.csv")
ARCHIVO_REPORTE_METRICAS = os.path.join(DIRECTORIO_RAIZ, "reports", "metricas_comparativas.txt")
ARCHIVO_FIGURA_CONFUSION = os.path.join(DIRECTORIO_RAIZ, "reports", "figures", "matrices_confusion_lsp.png")
ARCHIVO_FIGURA_F1 = os.path.join(DIRECTORIO_RAIZ, "reports", "figures", "comparativa_f1_por_clase.png")
ARCHIVO_MODELO_SERIALIZADO = os.path.join(DIRECTORIO_RAIZ, "models", "modelo_lsp_ganador.joblib")

PROPORCION_CONJUNTO_PRUEBA = 0.20
SEMILLA_ALEATORIA = 42
REPETICIONES_MEDICION_LATENCIA = 300
RESOLUCION_DPI_GRAFICAS = 300
COLUMNAS_METADATOS: list[str] = ["label", "class_name", "integrante"]


@dataclass
class ResultadoEvaluacion:
    """Estructura contenedora para registrar las métricas empíricas de cada modelo."""
    nombre_modelo: str
    modelo_entrenado: Any
    tiempo_entrenamiento_seg: float
    latencia_inferencia_ms: float
    exactitud_global: float
    f1_macro: float
    f1_weighted: float
    reporte_clasificacion: dict[str, Any]
    matriz_confusion_normalizada: np.ndarray
    f1_por_clase: list[float]
    tamano_estimado_mb: float


def cargar_y_validar_dataset(ruta_csv: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Carga el dataset consolidado, valida integridad y extrae matrices de diseno."""
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo del dataset consolidado: '{ruta_csv}'")

    print(f"Cargando dataset desde: '{ruta_csv}'...")
    df = pd.read_csv(ruta_csv)

    # Validar que no existan registros con valores faltantes
    nulos_totales = df.isnull().sum().sum()
    if nulos_totales > 0:
        print(f"Advertencia: Se detectaron {nulos_totales} valores nulos. Limpiando filas...")
        df = df.dropna()

    # Identificar columnas de características geométricas (excluyendo metadatos)
    columnas_caracteristicas = [col for col in df.columns if col not in COLUMNAS_METADATOS]
    total_caracteristicas = len(columnas_caracteristicas)
    print(f"Dataset cargado con éxito: {df.shape[0]} muestras y {total_caracteristicas} características por muestra.")

    # Mapeo consistente de clases asegurando orden ascendente de etiqueta
    df_clases = df[["label", "class_name"]].drop_duplicates().sort_values("label") # Verificacion de duplicados
    nombres_clases = df_clases["class_name"].tolist()

    print("\nBalance del Dataset por Clase:")
    conteo_clases = df.groupby(["label", "class_name"]).size()
    for (etiqueta, nombre), cantidad in conteo_clases.items():
        porcentaje = (cantidad / len(df)) * 100
        print(f"  - Clase [{etiqueta}] {nombre:<12}: {cantidad:>5} muestras ({porcentaje:.2f}%)")

    matriz_x = df[columnas_caracteristicas].to_numpy(dtype=np.float32)
    vector_y = df["label"].to_numpy(dtype=np.int64)

    return matriz_x, vector_y, nombres_clases, columnas_caracteristicas


def particionar_datos_estratificados(
    matriz_x: np.ndarray,
    vector_y: np.ndarray,
    proporcion_prueba: float,
    semilla: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genera una partición estratificada manteniendo la proporción idéntica de clases."""
    print(f"\nParticionando datos: {(1 - proporcion_prueba)*100:.0f}% Entrenamiento / {proporcion_prueba*100:.0f}% Prueba...")
    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
        matriz_x,
        vector_y,
        test_size=proporcion_prueba,
        stratify=vector_y,
        random_state=semilla
    )
    print(f"Conjunto de Entrenamiento: {x_entrenamiento.shape[0]} muestras.")
    print(f"Conjunto de Evaluación:    {x_prueba.shape[0]} muestras.")
    return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba


def construir_modelos_candidatos(semilla: int) -> dict[str, Any]:
    """Instancia los modelos de clasificación a comparar dentro de sus respectivos pipelines."""
    # Pipeline SVM con escalado estándar para evitar fuga de datos (data leakage)
    svm_pipeline = Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=semilla
        ))
    ])

    # Random Forest con paralelismo activado para acelerar convergencia
    bosque_aleatorio = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        criterion="gini",
        random_state=semilla,
        n_jobs=-1
    )

    return {
        "Support Vector Machine (RBF)": svm_pipeline,
        "Random Forest Classifier": bosque_aleatorio
    }


def medir_latencia_promedio_ms(modelo: Any, x_prueba: np.ndarray, repeticiones: int) -> float:
    """Mide empíricamente la latencia promedio por muestra en milisegundos para inferencia en tiempo real."""
    # Selección de un subconjunto aleatorio de muestras de prueba
    indices = np.random.choice(len(x_prueba), size=min(repeticiones, len(x_prueba)), replace=False)
    muestras = x_prueba[indices]

    # Precalentamiento del estimador (warm-up)
    _ = modelo.predict(muestras[:1])

    tiempos: list[float] = []
    for i in range(len(muestras)):
        muestra_individual = muestras[i : i + 1]
        inicio = time.perf_counter()
        _ = modelo.predict(muestra_individual)
        fin = time.perf_counter()
        tiempos.append((fin - inicio) * 1000.0)  # Conversión a milisegundos

    return float(np.mean(tiempos))


def estimar_tamano_modelo_mb(modelo: Any) -> float:
    """Calcula el tamano serializado del modelo en memoria para evaluar huella en el robot NAO."""
    buffer = io.BytesIO()
    joblib.dump(modelo, buffer)
    tamano_bytes = buffer.getbuffer().nbytes
    return tamano_bytes / (1024.0 * 1024.0)


def entrenar_y_evaluar_modelo(
    nombre_modelo: str,
    estimador: Any,
    x_entrenamiento: np.ndarray,
    y_entrenamiento: np.ndarray,
    x_prueba: np.ndarray,
    y_prueba: np.ndarray,
    nombres_clases: list[str]
) -> ResultadoEvaluacion:
    """Ejecuta el ciclo completo de ajuste, cronometraje y evaluación métrica de un estimador."""
    print(f"\n--- Iniciando entrenamiento: {nombre_modelo} ---")
    
    tiempo_inicio = time.perf_counter()
    estimador.fit(x_entrenamiento, y_entrenamiento)
    tiempo_fin = time.perf_counter()
    tiempo_entrenamiento = tiempo_fin - tiempo_inicio
    print(f"Entrenamiento completado en {tiempo_entrenamiento:.3f} segundos.")

    # Inferencia sobre el conjunto de evaluación
    predicciones = estimador.predict(x_prueba)

    # Cálculo de métricas principales
    exactitud = accuracy_score(y_prueba, predicciones)
    f1_macro = f1_score(y_prueba, predicciones, average="macro")
    f1_weighted = f1_score(y_prueba, predicciones, average="weighted")

    # Reporte detallado de clasificación
    reporte_dict = classification_report(
        y_prueba,
        predicciones,
        target_names=nombres_clases,
        output_dict=True
    )

    f1_por_clase = [reporte_dict[clase]["f1-score"] for clase in nombres_clases]

    # Matriz de confusión normalizada (proporciones sobre el total real por fila)
    matriz_conf = confusion_matrix(y_prueba, predicciones, normalize="true")

    latencia_ms = medir_latencia_promedio_ms(estimador, x_prueba, REPETICIONES_MEDICION_LATENCIA)
    tamano_mb = estimar_tamano_modelo_mb(estimador)

    print(f"Exactitud (Accuracy): {exactitud * 100:.2f}% | F1-Macro: {f1_macro * 100:.2f}%")
    print(f"Latencia de inferencia: {latencia_ms:.3f} ms/muestra | Tamano: {tamano_mb:.2f} MB")

    return ResultadoEvaluacion(
        nombre_modelo=nombre_modelo,
        modelo_entrenado=estimador,
        tiempo_entrenamiento_seg=tiempo_entrenamiento,
        latencia_inferencia_ms=latencia_ms,
        exactitud_global=exactitud,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        reporte_clasificacion=reporte_dict,
        matriz_confusion_normalizada=matriz_conf,
        f1_por_clase=f1_por_clase,
        tamano_estimado_mb=tamano_mb
    )


def generar_archivo_reporte_metricas(
    resultados: list[ResultadoEvaluacion],
    nombres_clases: list[str],
    ruta_salida: str
) -> None:
    """Exporta las métricas tabuladas en formato texto estructurado para el informe técnico."""
    lineas: list[str] = []
    separador = "=" * 80
    lineas.append(separador)
    lineas.append("INFORME COMPARATIVO DE RENDIMIENTO DE MODELOS - LENGUA DE SENAS PERUANA (LSP)")
    lineas.append("Proyecto: Clasificación de Gestos en Tiempo Real para el Robot NAO")
    lineas.append(separador)

    # Tabla resumen general
    lineas.append("\n1. COMPARATIVA GENERAL DE MODELOS")
    lineas.append("-" * 80)
    lineas.append(f"{'Modelo':<32} | {'Accuracy':<10} | {'F1-Macro':<10} | {'F1-Weight':<10} | {'Latencia':<10} | {'Entrenamiento'}")
    lineas.append("-" * 80)
    for res in resultados:
        lineas.append(
            f"{res.nombre_modelo:<32} | "
            f"{res.exactitud_global * 100:>8.2f}% | "
            f"{res.f1_macro * 100:>8.2f}% | "
            f"{res.f1_weighted * 100:>8.2f}% | "
            f"{res.latencia_inferencia_ms:>7.3f} ms | "
            f"{res.tiempo_entrenamiento_seg:>7.2f} s"
        )
    lineas.append("-" * 80)

    # Desglose de F1-Score por clase
    lineas.append("\n2. COMPARATIVA DE F1-SCORE POR CLASE")
    lineas.append("-" * 80)
    encabezado_clases = f"{'Clase':<16} | " + " | ".join([f"{r.nombre_modelo[:20]:<20}" for r in resultados])
    lineas.append(encabezado_clases)
    lineas.append("-" * 80)

    for i, clase in enumerate(nombres_clases):
        valores_clase = " | ".join([f"{r.f1_por_clase[i] * 100:>18.2f}%" for r in resultados])
        lineas.append(f"{clase:<16} | {valores_clase}")
    lineas.append("-" * 80)

    # Desglose completo de Precision, Recall y Soporte por modelo
    lineas.append("\n3. REPORTES DETALLADOS DE CLASIFICACIÓN")
    lineas.append(separador)
    for res in resultados:
        lineas.append(f"\n>>> Modelo: {res.nombre_modelo}")
        lineas.append(f"{'Clase':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Soporte'}")
        lineas.append("-" * 65)
        for clase in nombres_clases:
            metricas = res.reporte_clasificacion[clase]
            lineas.append(
                f"{clase:<16} | "
                f"{metricas['precision'] * 100:>8.2f}% | "
                f"{metricas['recall'] * 100:>8.2f}% | "
                f"{metricas['f1-score'] * 100:>8.2f}% | "
                f"{int(metricas['support']):>7}"
            )
        lineas.append("-" * 65)

    contenido = "\n".join(lineas)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(contenido)

    print(f"\nArchivo de reporte de métricas guardado en: '{ruta_salida}'")


def generar_grafica_matrices_confusion(
    resultados: list[ResultadoEvaluacion],
    nombres_clases: list[str],
    ruta_figura: str
) -> None:
    """Genera y guarda una figura enfrentada de matrices de confusión con resolución editorial (300 DPI)."""
    fig, ejes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    paletas = ["Blues", "Greens"]

    for idx, res in enumerate(resultados):
        eje = ejes[idx]
        matriz_pct = res.matriz_confusion_normalizada * 100.0

        sns.heatmap(
            matriz_pct,
            annot=True,
            fmt=".1f",
            cmap=paletas[idx % len(paletas)],
            cbar=True,
            ax=eje,
            xticklabels=nombres_clases,
            yticklabels=nombres_clases,
            vmin=0.0,
            vmax=100.0,
            annot_kws={"size": 9, "weight": "bold"}
        )

        eje.set_title(
            f"{res.nombre_modelo}\nExactitud Global: {res.exactitud_global * 100:.2f}% | F1-Macro: {res.f1_macro * 100:.2f}%",
            fontsize=12,
            pad=12,
            fontweight="semibold"
        )
        eje.set_xlabel("Clase Predicha", fontsize=11, fontweight="medium")
        if idx == 0:
            eje.set_ylabel("Clase Verdadera", fontsize=11, fontweight="medium")
        eje.tick_params(axis="x", rotation=35)
        eje.tick_params(axis="y", rotation=0)

    plt.suptitle(
        "Matrices de Confusión Normalizadas (%) - Clasificación de Lengua de Senas Peruana (LSP)",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=RESOLUCION_DPI_GRAFICAS, bbox_inches="tight")
    plt.close()
    print(f"Gráfica de matrices de confusión guardada en: '{ruta_figura}' (300 DPI).")


def generar_grafica_comparativa_f1(
    resultados: list[ResultadoEvaluacion],
    nombres_clases: list[str],
    ruta_figura: str
) -> None:
    """Genera una gráfica de barras comparando el desempeno clase por clase entre ambos modelos."""
    plt.figure(figsize=(13, 6))

    numero_clases = len(nombres_clases)
    indices = np.arange(numero_clases)
    ancho_barra = 0.38

    colores = ["#1f77b4", "#2ca02c"]

    for idx, res in enumerate(resultados):
        posiciones = indices - ancho_barra / 2 + idx * ancho_barra
        f1_porcentajes = [valor * 100.0 for valor in res.f1_por_clase]

        barras = plt.bar(
            posiciones,
            f1_porcentajes,
            width=ancho_barra,
            label=f"{res.nombre_modelo} (Promedio: {res.f1_macro * 100:.2f}%)",
            color=colores[idx % len(colores)],
            edgecolor="black",
            alpha=0.88
        )

        # Anotación del valor exacto sobre cada barra
        for barra in barras:
            altura = barra.get_height()
            plt.annotate(
                f"{altura:.1f}%",
                xy=(barra.get_x() + barra.get_width() / 2, altura),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                fontweight="semibold"
            )

    plt.title(
        "Comparativa de F1-Score por Clase de Sena (SVM vs Random Forest)",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    plt.xlabel("Clases de Senas (LSP)", fontsize=11, fontweight="medium")
    plt.ylabel("F1-Score (%)", fontsize=11, fontweight="medium")
    plt.xticks(indices, nombres_clases, fontsize=10, fontweight="medium")
    plt.ylim(0, 110)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=RESOLUCION_DPI_GRAFICAS, bbox_inches="tight")
    plt.close()
    print(f"Gráfica comparativa de F1-Score guardada en: '{ruta_figura}' (300 DPI).")


def serializar_mejor_modelo(
    resultados: list[ResultadoEvaluacion],
    nombres_clases: list[str],
    columnas_caracteristicas: list[str],
    ruta_salida: str
) -> ResultadoEvaluacion:
    """Selecciona el modelo con mejor F1-Macro y lo exporta empaquetado con sus metadatos de producción."""
    # Selección basada en F1-Macro para favorecer el balance equitativo de todas las clases
    mejor_resultado = max(resultados, key=lambda r: r.f1_macro)

    print("\n" + "=" * 80)
    print(f"MODELO GANADOR SELECCIONADO: {mejor_resultado.nombre_modelo}")
    print(f"Criterio: Mayor F1-Score Macro ({mejor_resultado.f1_macro * 100:.2f}%)")
    print(f"Latencia promedio: {mejor_resultado.latencia_inferencia_ms:.3f} ms/muestra")
    print("=" * 80)

    # Paquete de producción que incluye metadatos críticos para inferencia en el NAO
    paquete_produccion: dict[str, Any] = {
        "modelo": mejor_resultado.modelo_entrenado,
        "nombre_modelo": mejor_resultado.nombre_modelo,
        "clases": nombres_clases,
        "columnas_caracteristicas": columnas_caracteristicas,
        "metricas": {
            "exactitud_global": mejor_resultado.exactitud_global,
            "f1_macro": mejor_resultado.f1_macro,
            "f1_weighted": mejor_resultado.f1_weighted,
            "latencia_inferencia_ms": mejor_resultado.latencia_inferencia_ms,
            "tiempo_entrenamiento_seg": mejor_resultado.tiempo_entrenamiento_seg
        }
    }

    joblib.dump(paquete_produccion, ruta_salida)
    print(f"Modelo ganador empaquetado y serializado exitosamente en: '{ruta_salida}'")
    return mejor_resultado


def main() -> None:
    """Punto de entrada principal para el pipeline de experimentación y entrenamiento."""
    print("=" * 80)
    print("PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN COMPARATIVA DE MODELOS (LSP - ROBOT NAO)")
    print("=" * 80)

    # 1. Carga, exploración y extracción
    matriz_x, vector_y, nombres_clases, columnas_caracteristicas = cargar_y_validar_dataset(
        ARCHIVO_DATASET_CONSOLIDADO
    )

    # 2. Partición estratificada (80% / 20%)
    x_train, x_test, y_train, y_test = particionar_datos_estratificados(
        matriz_x,
        vector_y,
        PROPORCION_CONJUNTO_PRUEBA,
        SEMILLA_ALEATORIA
    )

    # 3. Modelos candidatos
    modelos = construir_modelos_candidatos(SEMILLA_ALEATORIA)

    # 4. Ajuste, evaluación y registro métrico
    resultados: list[ResultadoEvaluacion] = []
    for nombre, estimador in modelos.items():
        resultado = entrenar_y_evaluar_modelo(
            nombre,
            estimador,
            x_train,
            y_train,
            x_test,
            y_test,
            nombres_clases
        )
        resultados.append(resultado)

    # 5. Exportación de métricas a archivo de texto
    generar_archivo_reporte_metricas(
        resultados,
        nombres_clases,
        ARCHIVO_REPORTE_METRICAS
    )

    # 6. Generación de figuras en alta resolución (300 DPI)
    generar_grafica_matrices_confusion(
        resultados,
        nombres_clases,
        ARCHIVO_FIGURA_CONFUSION
    )

    generar_grafica_comparativa_f1(
        resultados,
        nombres_clases,
        ARCHIVO_FIGURA_F1
    )

    # 7. Serialización del modelo ganador para despliegue
    serializar_mejor_modelo(
        resultados,
        nombres_clases,
        columnas_caracteristicas,
        ARCHIVO_MODELO_SERIALIZADO
    )

    print("\n¡Pipeline de Machine Learning finalizado con éxito!")
    print(f"1. Métricas tabuladas:  '{ARCHIVO_REPORTE_METRICAS}'")
    print(f"2. Matriz de confusión: '{ARCHIVO_FIGURA_CONFUSION}'")
    print(f"3. Comparativa F1:      '{ARCHIVO_FIGURA_F1}'")
    print(f"4. Modelo listo en:     '{ARCHIVO_MODELO_SERIALIZADO}'")


if __name__ == "__main__":
    main()
