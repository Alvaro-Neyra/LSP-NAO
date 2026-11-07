import os
import re
import time
from typing import Any
import cv2
import mediapipe as mp
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Diccionario de clases para Lengua de Señas Peruana (LSP)
CLASES: dict[int, str] = {
    0: "YO",
    1: "TU",
    2: "NECESITAR",
    3: "QUERER",
    4: "AGUA",
    5: "AYUDA",
    6: "MEDICINA",
    7: "ENVIAR"
}

# Constantes de balance y muestreo colaborativo
LIMITE_MUESTRAS_POR_CLASE = 300
TOTAL_CLASES = len(CLASES)
TOTAL_MUESTRAS_OBJETIVO = LIMITE_MUESTRAS_POR_CLASE * TOTAL_CLASES  # 2400 muestras por integrante
INTERVALO_MUESTREO_SEGUNDOS = 0.04  # ~25 muestras por segundo
CONFIANZA_MINIMA_DETECCION = 0.7
CONFIANZA_MINIMA_SEGUIMIENTO = 0.5
TOTAL_PUNTOS_POR_MANO = 21
DIMENSIONES_POR_PUNTO = 3
TOTAL_VALORES_POR_MANO = TOTAL_PUNTOS_POR_MANO * DIMENSIONES_POR_PUNTO  # 63 valores por mano


def solicitar_identificador_integrante() -> tuple[str, str]:
    """Solicita el nombre del integrante y genera la ruta del archivo CSV personalizado."""
    print("=" * 60)
    print("SISTEMA DE RECOLECCIÓN COLABORATIVA DE DATASET LSP")
    print(f"Meta por integrante: {LIMITE_MUESTRAS_POR_CLASE} muestras por clase ({TOTAL_MUESTRAS_OBJETIVO} en total)")
    print("=" * 60)
    
    entrada = input("Ingrese su nombre o alias (ej. alvaro, victor, rai, arturo): ").strip().lower()
    # Limpieza de caracteres no alfanuméricos para evitar nombres inválidos de archivo
    nombre_limpio = re.sub(r"[^a-z0-9_]", "", entrada) if entrada else "integrante"
    
    directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    carpeta_destino = os.path.join(directorio_raiz, "data", "raw")
    os.makedirs(carpeta_destino, exist_ok=True)
    ruta_archivo = os.path.join(carpeta_destino, f"dataset_lsp_{nombre_limpio}.csv")

    # Si existe un dataset genérico previo y el integrante es el titular, permite adoptarlo
    if not os.path.exists(ruta_archivo) and os.path.exists("dataset_lsp.csv"):
        try:
            os.replace("dataset_lsp.csv", ruta_archivo)
            print(f"Se migró el progreso existente de 'dataset_lsp.csv' a '{ruta_archivo}'.")
        except Exception:
            pass

    return nombre_limpio, ruta_archivo


def generar_encabezados_csv() -> list[str]:
    """Genera las 126 columnas de landmarks junto a las etiquetas de clasificación."""
    ejes = ["x", "y", "z"]
    columnas_izquierda = [
        f"lh_lm_{i}_{eje}" for i in range(TOTAL_PUNTOS_POR_MANO) for eje in ejes
    ]
    columnas_derecha = [
        f"rh_lm_{i}_{eje}" for i in range(TOTAL_PUNTOS_POR_MANO) for eje in ejes
    ]
    return columnas_izquierda + columnas_derecha + ["label", "class_name"]


def inicializar_archivo_csv(ruta_csv: str, encabezados: list[str]) -> None:
    """Crea el archivo CSV del integrante con sus encabezados si no existe."""
    if not os.path.exists(ruta_csv):
        df_nuevo = pd.DataFrame(columns=encabezados)
        df_nuevo.to_csv(ruta_csv, index=False)
        print(f"Archivo {ruta_csv} inicializado correctamente.")


def cargar_conteo_existente(ruta_csv: str) -> dict[int, int]:
    """Calcula cuántas muestras existen ya registradas para cada clase en el CSV."""
    conteo = {idx: 0 for idx in CLASES.keys()}
    if not os.path.exists(ruta_csv):
        return conteo

    try:
        df = pd.read_csv(ruta_csv)
        if "label" in df.columns:
            conteos_series = df["label"].value_counts()
            for idx in conteo.keys():
                if idx in conteos_series.index:
                    conteo[idx] = int(conteos_series[idx])
    except Exception as error:
        print(f"Advertencia al leer progreso previo: {error}")

    return conteo


def normalizar_puntos_mano(puntos_mano: Any) -> list[float]:
    """Normaliza las coordenadas relativas a la muñeca y escala invariantemente."""
    punto_base = puntos_mano.landmark[0]
    base_x, base_y, base_z = punto_base.x, punto_base.y, punto_base.z

    coordenadas_relativas: list[float] = []
    for punto in puntos_mano.landmark:
        coordenadas_relativas.extend([
            punto.x - base_x,
            punto.y - base_y,
            punto.z - base_z
        ])

    distancia_maxima = max(max(map(abs, coordenadas_relativas)), 1e-6)
    return [valor / distancia_maxima for valor in coordenadas_relativas]


def extraer_vectores_manos(resultados: Any) -> tuple[list[float], list[float], bool, bool]:
    """Separa y normaliza las coordenadas correspondientes a cada mano."""
    vector_izq = [0.0] * TOTAL_VALORES_POR_MANO
    vector_der = [0.0] * TOTAL_VALORES_POR_MANO
    detectada_izq = False
    detectada_der = False

    if not resultados.multi_hand_landmarks or not resultados.multi_handedness:
        return vector_izq, vector_der, detectada_izq, detectada_der

    for puntos, clasificacion in zip(resultados.multi_hand_landmarks, resultados.multi_handedness):
        etiqueta_lateralidad = clasificacion.classification[0].label  # 'Left' o 'Right'
        vector_normalizado = normalizar_puntos_mano(puntos)

        if etiqueta_lateralidad == "Left":
            if not detectada_izq:
                vector_izq = vector_normalizado
                detectada_izq = True
            else:
                vector_der = vector_normalizado
                detectada_der = True
        elif etiqueta_lateralidad == "Right":
            if not detectada_der:
                vector_der = vector_normalizado
                detectada_der = True
            else:
                vector_izq = vector_normalizado
                detectada_izq = True

    return vector_izq, vector_der, detectada_izq, detectada_der


def buscar_siguiente_clase_pendiente(conteo_por_clase: dict[int, int], clase_actual: int) -> int | None:
    """Encuentra la siguiente clase que aún tenga menos de 300 muestras."""
    for paso in range(1, TOTAL_CLASES + 1):
        candidata = (clase_actual + paso) % TOTAL_CLASES
        if conteo_por_clase[candidata] < LIMITE_MUESTRAS_POR_CLASE:
            return candidata
    return None


def dibujar_interfaz(
    fotograma: Any,
    nombre_integrante: str,
    indice_clase: int,
    conteo_por_clase: dict[int, int],
    esta_grabando: bool,
    mano_izq: bool,
    mano_der: bool
) -> None:
    """Dibuja el panel HUD con métricas de progreso de clase y total."""
    muestras_clase = conteo_por_clase[indice_clase]
    total_muestras = sum(conteo_por_clase.values())
    porcentaje_total = (total_muestras / TOTAL_MUESTRAS_OBJETIVO) * 100

    cv2.rectangle(fotograma, (10, 10), (470, 195), (20, 20, 20), -1)

    # Estado de la grabación
    if muestras_clase >= LIMITE_MUESTRAS_POR_CLASE:
        estado_texto = "¡CLASE COMPLETADA (300/300)!"
        color_estado = (0, 215, 255)  # Dorado / Amarillo
    elif esta_grabando:
        estado_texto = "GRABANDO..."
        color_estado = (0, 255, 0)    # Verde
    else:
        estado_texto = "PAUSADO"
        color_estado = (0, 0, 255)    # Rojo

    cv2.putText(fotograma, f"Integrante: {nombre_integrante.upper()}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 200), 2)
    cv2.putText(fotograma, f"Clase [{indice_clase}]: {CLASES[indice_clase]} ({muestras_clase}/{LIMITE_MUESTRAS_POR_CLASE})", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(fotograma, f"Progreso total: {total_muestras}/{TOTAL_MUESTRAS_OBJETIVO} ({porcentaje_total:.1f}%)", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 2)
    cv2.putText(fotograma, f"Estado: {estado_texto}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_estado, 2)

    texto_manos = f"Izq: {'SI' if mano_izq else 'NO'} | Der: {'SI' if mano_der else 'NO'}"
    cv2.putText(fotograma, texto_manos, (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # Notificación central si el integrante completó el dataset total
    if total_muestras >= TOTAL_MUESTRAS_OBJETIVO:
        cv2.rectangle(fotograma, (40, 220), (600, 270), (0, 150, 0), -1)
        cv2.putText(fotograma, "¡META COMPLETA! LISTO PARA ENVIAR", (50, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def main() -> None:
    """Controla el flujo de captura con límite por clase y gestión colaborativa."""
    nombre_integrante, ruta_csv = solicitar_identificador_integrante()
    encabezados = generar_encabezados_csv()
    inicializar_archivo_csv(ruta_csv, encabezados)
    conteo_por_clase = cargar_conteo_existente(ruta_csv)

    # Selecciona la primera clase que aún esté incompleta
    indice_clase_actual = 0
    if conteo_por_clase[0] >= LIMITE_MUESTRAS_POR_CLASE:
        pendiente = buscar_siguiente_clase_pendiente(conteo_por_clase, 0)
        indice_clase_actual = pendiente if pendiente is not None else 0

    detector_manos = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=CONFIANZA_MINIMA_DETECCION,
        min_tracking_confidence=CONFIANZA_MINIMA_SEGUIMIENTO
    )
    utilidad_dibujo = mp.solutions.drawing_utils

    captura = cv2.VideoCapture(0)
    esta_grabando = False

    print("\nControles:")
    print("  - Teclas [0 - 7] : Cambiar clase a registrar.")
    print("  - Barra Espacio  : Iniciar / Pausar grabación.")
    print("  - Tecla [Q]      : Salir y guardar progreso.")
    print("=" * 60)

    try:
        while captura.isOpened():
            exito, fotograma = captura.read()
            if not exito:
                break

            fotograma = cv2.flip(fotograma, 1)
            fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
            resultados = detector_manos.process(fotograma_rgb)

            if resultados.multi_hand_landmarks:
                for marcas_mano in resultados.multi_hand_landmarks:
                    utilidad_dibujo.draw_landmarks(
                        fotograma, marcas_mano, mp.solutions.hands.HAND_CONNECTIONS
                    )

            vec_izq, vec_der, detectada_izq, detectada_der = extraer_vectores_manos(resultados)

            # Lógica de grabación con límite estricto de 300 muestras
            if esta_grabando and (detectada_izq or detectada_der):
                if conteo_por_clase[indice_clase_actual] < LIMITE_MUESTRAS_POR_CLASE:
                    vector_completo = vec_izq + vec_der + [indice_clase_actual, CLASES[indice_clase_actual]]
                    pd.DataFrame([vector_completo]).to_csv(
                        ruta_csv, mode="a", header=False, index=False
                    )
                    conteo_por_clase[indice_clase_actual] += 1
                    time.sleep(INTERVALO_MUESTREO_SEGUNDOS)

                    # Si alcanza exactamente el límite, pausa y busca la siguiente clase
                    if conteo_por_clase[indice_clase_actual] >= LIMITE_MUESTRAS_POR_CLASE:
                        esta_grabando = False
                        print(f"\n¡Clase [{indice_clase_actual}] ({CLASES[indice_clase_actual]}) completada (300/300)!")
                        siguiente = buscar_siguiente_clase_pendiente(conteo_por_clase, indice_clase_actual)
                        if siguiente is not None:
                            indice_clase_actual = siguiente
                            print(f"Avanzando a la clase pendiente [{indice_clase_actual}]: {CLASES[indice_clase_actual]}")
                        else:
                            print("¡FELICITACIONES! Has completado las 8 clases (2400 muestras en total).")
                else:
                    esta_grabando = False

            dibujar_interfaz(
                fotograma,
                nombre_integrante,
                indice_clase_actual,
                conteo_por_clase,
                esta_grabando,
                detectada_izq,
                detectada_der
            )

            cv2.imshow(f"Captura LSP - {nombre_integrante.upper()}", fotograma)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            elif tecla == 32:  # Barra espaciadora
                if conteo_por_clase[indice_clase_actual] < LIMITE_MUESTRAS_POR_CLASE:
                    esta_grabando = not esta_grabando
                else:
                    print(f"La clase [{indice_clase_actual}] ya tiene 300 muestras. Selecciona otra clase.")
            elif ord("0") <= tecla <= ord("7"):
                nueva_clase = tecla - ord("0")
                indice_clase_actual = nueva_clase
                esta_grabando = False
                print(f"Seleccionada clase {indice_clase_actual}: {CLASES[indice_clase_actual]} ({conteo_por_clase[indice_clase_actual]}/{LIMITE_MUESTRAS_POR_CLASE})")

    finally:
        captura.release()
        cv2.destroyAllWindows()
        detector_manos.close()


if __name__ == "__main__":
    main()
