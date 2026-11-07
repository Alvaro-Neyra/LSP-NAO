import glob
import os
import re
import pandas as pd

# Constantes de configuración para la unificación
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIRECTORIO_DATOS_RAW = os.path.join(DIRECTORIO_RAIZ, "data", "raw")
DIRECTORIO_DATOS_PROCESSED = os.path.join(DIRECTORIO_RAIZ, "data", "processed")

PATRON_BUSQUEDA_CSV = os.path.join(DIRECTORIO_DATOS_RAW, "dataset_lsp_*.csv")
ARCHIVO_SALIDA_CONSOLIDADO = os.path.join(DIRECTORIO_DATOS_PROCESSED, "dataset_lsp_consolidado.csv")
PATRONES_EXCLUIDOS: list[str] = ["backup", "consolidado", "unimanual"]
COLUMNA_METADATO_INTEGRANTE = "integrante"
TOTAL_COLUMNAS_BASE_ESPERADAS = 128  # 126 coordenadas + label + class_name


def obtener_archivos_validos(patron: str, exclusiones: list[str]) -> list[str]:
    """Busca archivos CSV que coincidan con el patrón descartando respaldos y consolidados."""
    archivos_encontrados = glob.glob(patron)
    archivos_filtrados: list[str] = []

    for ruta in archivos_encontrados:
        nombre_base = os.path.basename(ruta).lower()
        # Descarta archivos temporales o respaldos previos
        if not any(exclusion in nombre_base for exclusion in exclusiones):
            archivos_filtrados.append(ruta)

    return sorted(archivos_filtrados)


def extraer_nombre_integrante(ruta_archivo: str) -> str:
    """Extrae el alias o nombre del integrante a partir del nombre del archivo."""
    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]
    coincidencia = re.sub(r"^dataset_lsp_", "", nombre_archivo)
    return coincidencia if coincidencia else "desconocido"


def procesar_archivo_individual(ruta_archivo: str, nombre_integrante: str) -> pd.DataFrame | None:
    """Lee y valida un dataset individual agregando la columna de trazabilidad."""
    try:
        df = pd.read_csv(ruta_archivo)
        
        # Validar consistencia de dimensiones
        if df.shape[1] != TOTAL_COLUMNAS_BASE_ESPERADAS:
            print(f"Advertencia: '{ruta_archivo}' tiene {df.shape[1]} columnas (se esperaban {TOTAL_COLUMNAS_BASE_ESPERADAS}). Omitido.")
            return None

        # Comprobar presencia de valores nulos
        nulos_totales = df.isnull().sum().sum()
        if nulos_totales > 0:
            print(f"Advertencia: '{ruta_archivo}' contiene {nulos_totales} valores nulos. Limpiando filas...")
            df = df.dropna()

        # Añadir columna con el nombre del integrante como metadato
        df[COLUMNA_METADATO_INTEGRANTE] = nombre_integrante
        print(f"Procesado: {ruta_archivo} -> {len(df)} muestras cargadas (Integrante: {nombre_integrante}).")
        return df

    except Exception as error:
        print(f"Error al leer '{ruta_archivo}': {error}")
        return None


def mostrar_resumen_estadistico(df_consolidado: pd.DataFrame) -> None:
    """Imprime en consola un informe con el balance general de clases e integrantes."""
    total_muestras = len(df_consolidado)
    print("\n" + "=" * 65)
    print(f"RESUMEN DEL DATASET CONSOLIDADO (Total: {total_muestras} muestras)")
    print("=" * 65)

    print("\nDistribución por Integrante:")
    conteo_integrantes = df_consolidado[COLUMNA_METADATO_INTEGRANTE].value_counts()
    for integrante, cantidad in conteo_integrantes.items():
        porcentaje = (cantidad / total_muestras) * 100
        print(f"  - {integrante:<15}: {cantidad:>5} muestras ({porcentaje:.1f}%)")

    print("\nDistribución por Clase (Lengua de Señas Peruana):")
    conteo_clases = df_consolidado.groupby(["label", "class_name"]).size()
    for (etiqueta, nombre_clase), cantidad in conteo_clases.items():
        porcentaje = (cantidad / total_muestras) * 100
        print(f"  - [{etiqueta}] {nombre_clase:<12}: {cantidad:>5} muestras ({porcentaje:.1f}%)")
    print("=" * 65)


def unificar_datasets() -> None:
    """Función principal que coordina la lectura, validación y unión de datasets."""
    print("Iniciando búsqueda de datasets individuales...")
    rutas_archivos = obtener_archivos_validos(PATRON_BUSQUEDA_CSV, PATRONES_EXCLUIDOS)

    if not rutas_archivos:
        print("No se encontraron archivos de dataset válidos para consolidar.")
        return

    dataframes_validos: list[pd.DataFrame] = []
    for ruta in rutas_archivos:
        integrante = extraer_nombre_integrante(ruta)
        df_individual = procesar_archivo_individual(ruta, integrante)
        if df_individual is not None:
            dataframes_validos.append(df_individual)

    if not dataframes_validos:
        print("No se pudo procesar ningún dataset válido.")
        return

    # Concatenar horizontalmente respetando el orden de columnas
    df_consolidado = pd.concat(dataframes_validos, ignore_index=True)

    # Exportar a archivo CSV maestro
    df_consolidado.to_csv(ARCHIVO_SALIDA_CONSOLIDADO, index=False)
    print(f"\nArchivo maestro generado exitosamente: '{ARCHIVO_SALIDA_CONSOLIDADO}'")
    
    mostrar_resumen_estadistico(df_consolidado)


if __name__ == "__main__":
    unificar_datasets()
