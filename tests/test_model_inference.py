"""Pruebas automatizadas de inferencia y latencia para el modelo ganador."""

import os
import time
import joblib
import numpy as np
import pytest

RUTA_MODELO = "models/modelo_lsp_ganador.joblib"
LATENCIA_MAXIMA_MS = 5.0
CARACTERISTICAS_ESPERADAS = 126


def test_archivo_modelo_existe() -> None:
    """Valida la existencia del artefacto serializado del modelo ganador."""
    assert os.path.exists(RUTA_MODELO), f"El archivo '{RUTA_MODELO}' no existe."


def test_inferencia_dummy_y_estructura() -> None:
    """Verifica que el modelo pueda inferir vectores sintéticos sin arrojar excepciones."""
    paquete = joblib.load(RUTA_MODELO)
    assert "modelo" in paquete, "El paquete no contiene la clave 'modelo'."
    assert "clases" in paquete, "El paquete no contiene la clave 'clases'."
    assert len(paquete["clases"]) == 8, "El número de clases registradas no es 8."

    modelo = paquete["modelo"]
    # Prueba con vector sintético de ceros (gesto nulo o mano ausente)
    vector_prueba = np.zeros((1, CARACTERISTICAS_ESPERADAS), dtype=np.float32)
    prediccion = modelo.predict(vector_prueba)
    assert 0 <= int(prediccion[0]) <= 7, f"Predicción fuera de rango: {prediccion[0]}"


def test_latencia_inferencia_en_tiempo_real() -> None:
    """Garantiza que la inferencia por muestra cumpla el umbral crítico para robótica (<5 ms)."""
    paquete = joblib.load(RUTA_MODELO)
    modelo = paquete["modelo"]
    vector_prueba = np.random.uniform(-1.0, 1.0, size=(1, CARACTERISTICAS_ESPERADAS)).astype(np.float32)

    # Medición de 50 predicciones individuales
    tiempos_ms: list[float] = []
    for _ in range(50):
        inicio = time.perf_counter()
        _ = modelo.predict(vector_prueba)
        fin = time.perf_counter()
        tiempos_ms.append((fin - inicio) * 1000.0)

    latencia_promedio = float(np.mean(tiempos_ms))
    assert latencia_promedio < LATENCIA_MAXIMA_MS, f"Latencia excesiva: {latencia_promedio:.3f} ms (máximo permitido: {LATENCIA_MAXIMA_MS} ms)."
