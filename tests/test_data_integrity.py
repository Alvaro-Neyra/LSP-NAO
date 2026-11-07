"""Pruebas automatizadas de integridad para el dataset procesado de LSP."""

import os
import pandas as pd
import pytest

RUTA_DATASET = "data/processed/dataset_lsp_consolidado.csv"
TOTAL_CLASES_ESPERADAS = 8
COLUMNAS_MINIMAS_ESPERADAS = 128


def test_archivo_dataset_existe() -> None:
    """Verifica que el archivo consolidado esté presente en la ruta processed."""
    assert os.path.exists(RUTA_DATASET), f"El archivo '{RUTA_DATASET}' no fue encontrado."


def test_dimensiones_y_nulos() -> None:
    """Valida que no existan valores nulos y que el volumen sea superior a 10,000 muestras."""
    df = pd.read_csv(RUTA_DATASET)
    assert len(df) >= 10000, f"Volumen insuficiente: {len(df)} filas encontradas."
    assert df.shape[1] >= COLUMNAS_MINIMAS_ESPERADAS, f"Columnas insuficientes: {df.shape[1]}."
    assert df.isnull().sum().sum() == 0, "Se detectaron valores nulos en el dataset."


def test_balance_de_clases() -> None:
    """Comprueba que todas las 8 clases estén presentes y tengan al menos 1,200 muestras."""
    df = pd.read_csv(RUTA_DATASET)
    conteo_clases = df["label"].value_counts()
    assert len(conteo_clases) == TOTAL_CLASES_ESPERADAS, f"Se esperaban {TOTAL_CLASES_ESPERADAS} clases, pero hay {len(conteo_clases)}."
    for etiqueta, cantidad in conteo_clases.items():
        assert cantidad >= 1200, f"La clase {etiqueta} tiene menos de 1,200 muestras ({cantidad})."
