"""Punto de entrada principal y orquestador del proyecto PROYECTO-NAO.

Permite ejecutar interactivamente o mediante argumentos CLI los módulos de captura,
unificación, entrenamiento de modelos y ejecución de pruebas de calidad.
"""

import argparse
import os
import subprocess
import sys


def ejecutar_comando(comando: list[str], descripcion: str) -> None:
    """Ejecuta un subproceso con el entorno actual e informa de su estatus."""
    print(f"\n[INICIANDO] {descripcion}...")
    try:
        resultado = subprocess.run(comando, check=False)
        if resultado.returncode == 0:
            print(f"[COMPLETADO] {descripcion} finalizado con éxito.\n")
        else:
            print(f"[ERROR] {descripcion} finalizó con código de salida {resultado.returncode}.\n")
    except Exception as error:
        print(f"[EXCEPCIÓN] Error al ejecutar {descripcion}: {error}\n")


def iniciar_captura_datos() -> None:
    """Lanza el script de captura bimanual con MediaPipe."""
    ruta_script = os.path.join("src", "vision", "capturar_dataset_lsp.py")
    ejecutar_comando([sys.executable, ruta_script], "Captura de Dataset LSP")


def iniciar_unificacion_datos() -> None:
    """Lanza el script de fusión y balanceo de datasets individuales."""
    ruta_script = os.path.join("src", "pipeline", "unificar_datasets.py")
    ejecutar_comando([sys.executable, ruta_script], "Unificación de Datasets")


def iniciar_entrenamiento_modelos() -> None:
    """Lanza el pipeline de entrenamiento comparativo y evaluación de métricas."""
    ruta_script = os.path.join("src", "pipeline", "entrenar_evaluar_modelos.py")
    ejecutar_comando([sys.executable, ruta_script], "Entrenamiento y Evaluación de Modelos")


def ejecutar_pruebas_calidad() -> None:
    """Ejecuta la suite de pruebas unitarias y de latencia con pytest."""
    ejecutar_comando([sys.executable, "-m", "pytest", "tests"], "Pruebas Automatizadas de Calidad")


def iniciar_pruebas_en_vivo() -> None:
    """Lanza el banco de pruebas de inferencia en tiempo real con captura automatica."""
    ruta_script = os.path.join("src", "vision", "probar_modelo_en_vivo.py")
    ejecutar_comando([sys.executable, ruta_script], "Banco de Pruebas e Inferencia en Vivo")


def iniciar_traductor_nao() -> None:
    """Lanza la aplicacion de inferencia y traduccion integrada con el Robot NAO."""
    ruta_script = os.path.join("src", "robot", "app_inferencia_nao.py")
    ejecutar_comando([sys.executable, ruta_script], "Traductor LSP con Robot NAO")


def mostrar_menu_interactivo() -> None:
    """Despliega un menú en consola para seleccionar la operación deseada."""
    while True:
        print("=" * 65)
        print(" SISTEMA DE RECONOCIMIENTO DE LENGUA DE SENAS PERUANA - ROBOT NAO")
        print(" Práctica N.° 2 - Inteligencia Artificial (UPC)")
        print("=" * 65)
        print("  [1] Capturar datos de senas (MediaPipe Hands Bimanual)")
        print("  [2] Unificar datasets individuales de integrantes (data/raw)")
        print("  [3] Entrenar y evaluar modelos ML (SVM vs Random Forest)")
        print("  [4] Ejecutar pruebas automatizadas de integridad y latencia")
        print("  [5] Probar modelo en tiempo real y capturar evidencias (Webcam)")
        print("  [6] Iniciar traductor de senas integrado con Robot NAO")
        print("  [0] Salir")
        print("=" * 65)

        opcion = input("Seleccione una opción: ").strip()

        if opcion == "1":
            iniciar_captura_datos()
        elif opcion == "2":
            iniciar_unificacion_datos()
        elif opcion == "3":
            iniciar_entrenamiento_modelos()
        elif opcion == "4":
            ejecutar_pruebas_calidad()
        elif opcion == "5":
            iniciar_pruebas_en_vivo()
        elif opcion == "6":
            iniciar_traductor_nao()
        elif opcion == "0":
            print("Cerrando el sistema. ¡Hasta pronto!")
            break
        else:
            print("Opción no válida. Intente nuevamente.\n")


def main() -> None:
    """Punto de entrada con soporte para argumentos de línea de comandos."""
    analizador = argparse.ArgumentParser(
        description="Orquestador del Proyecto de Reconocimiento de Senas (LSP) para el Robot NAO."
    )
    analizador.add_argument("--capturar", action="store_true", help="Inicia la captura bimanual de senas.")
    analizador.add_argument("--unificar", action="store_true", help="Unifica los datasets individuales.")
    analizador.add_argument("--entrenar", action="store_true", help="Ejecuta el entrenamiento y evaluación de modelos.")
    analizador.add_argument("--test", action="store_true", help="Ejecuta las pruebas automatizadas con pytest.")
    analizador.add_argument("--probar", action="store_true", help="Inicia el banco de pruebas e inferencia en vivo.")
    analizador.add_argument("--nao", action="store_true", help="Inicia el traductor de senas integrado con Robot NAO.")

    argumentos = analizador.parse_args()

    if argumentos.capturar:
        iniciar_captura_datos()
    elif argumentos.unificar:
        iniciar_unificacion_datos()
    elif argumentos.entrenar:
        iniciar_entrenamiento_modelos()
    elif argumentos.test:
        ejecutar_pruebas_calidad()
    elif argumentos.probar:
        iniciar_pruebas_en_vivo()
    elif argumentos.nao:
        iniciar_traductor_nao()
    else:
        mostrar_menu_interactivo()


if __name__ == "__main__":
    main()
