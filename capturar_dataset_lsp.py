import os
import time
from typing import Any
import cv2
import mediapipe as mp
import pandas as pd

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

# Constantes de configuración y captura
ARCHIVO_SALIDA_CSV = "dataset_lsp.csv"
INTERVALO_MUESTREO_SEGUNDOS = 0.04  # Aproximadamente 25 cuadros por segundo
CONFIANZA_MINIMA_DETECCION = 0.7
CONFIANZA_MINIMA_SEGUIMIENTO = 0.5
TOTAL_PUNTOS_POR_MANO = 21
DIMENSIONES_POR_PUNTO = 3
TOTAL_VALORES_POR_MANO = TOTAL_PUNTOS_POR_MANO * DIMENSIONES_POR_PUNTO  # 63 valores por mano


def generar_encabezados_csv() -> list[str]:
    """Genera las columnas para mano izquierda (lh), derecha (rh) y etiquetas."""
    ejes = ["x", "y", "z"]
    columnas_izquierda = [
        f"lh_lm_{i}_{eje}" for i in range(TOTAL_PUNTOS_POR_MANO) for eje in ejes
    ]
    columnas_derecha = [
        f"rh_lm_{i}_{eje}" for i in range(TOTAL_PUNTOS_POR_MANO) for eje in ejes
    ]
    return columnas_izquierda + columnas_derecha + ["label", "class_name"]


def inicializar_archivo_csv(ruta_csv: str, encabezados: list[str]) -> None:
    """Valida la consistencia del archivo CSV o crea uno nuevo con respaldo seguro."""
    if os.path.exists(ruta_csv):
        try:
            df_existente = pd.read_csv(ruta_csv, nrows=1)
            if list(df_existente.columns) == encabezados:
                print(f"Archivo {ruta_csv} cargado correctamente con estructura bimanual.")
                return
            # Respaldo automático si la estructura previa era unimanual
            ruta_respaldo = "dataset_lsp_unimanual_backup.csv"
            os.replace(ruta_csv, ruta_respaldo)
            print(f"Esquema anterior incompatible respaldado en: {ruta_respaldo}")
        except Exception as error:
            print(f"Advertencia al leer CSV previo: {error}")

    df_nuevo = pd.DataFrame(columns=encabezados)
    df_nuevo.to_csv(ruta_csv, index=False)
    print(f"Archivo {ruta_csv} inicializado con {len(encabezados)} columnas.")


def normalizar_puntos_mano(puntos_mano: Any) -> list[float]:
    """Normaliza los 21 puntos respecto a la muñeca y escala invariantemente."""
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
    """Extrae y asigna las coordenadas normalizadas a mano izquierda y derecha."""
    vector_izq = [0.0] * TOTAL_VALORES_POR_MANO
    vector_der = [0.0] * TOTAL_VALORES_POR_MANO
    detectada_izq = False
    detectada_der = False

    if not resultados.multi_hand_landmarks or not resultados.multi_handedness:
        return vector_izq, vector_der, detectada_izq, detectada_der

    for puntos, clasificacion in zip(resultados.multi_hand_landmarks, resultados.multi_handedness):
        etiqueta_lateralidad = clasificacion.classification[0].label  # 'Left' o 'Right'
        vector_normalizado = normalizar_puntos_mano(puntos)

        # Asignación con control defensivo por si ambas manos reciben la misma etiqueta
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


def dibujar_interfaz(
    fotograma: Any,
    indice_clase: int,
    esta_grabando: bool,
    muestras_capturadas: int,
    mano_izq: bool,
    mano_der: bool
) -> None:
    """Muestra el panel HUD en pantalla con la información de captura y estado."""
    cv2.rectangle(fotograma, (10, 10), (420, 155), (20, 20, 20), -1)
    
    estado_texto = "GRABANDO..." if esta_grabando else "PAUSADO"
    color_estado = (0, 255, 0) if esta_grabando else (0, 0, 255)

    cv2.putText(fotograma, f"Clase [{indice_clase}]: {CLASES[indice_clase]}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(fotograma, f"Estado: {estado_texto}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_estado, 2)
    cv2.putText(fotograma, f"Muestras sesion: {muestras_capturadas}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 2)

    color_izq = (0, 255, 0) if mano_izq else (120, 120, 120)
    color_der = (0, 255, 0) if mano_der else (120, 120, 120)
    texto_manos = f"Izq: {'SI' if mano_izq else 'NO'} | Der: {'SI' if mano_der else 'NO'}"
    cv2.putText(fotograma, texto_manos, (20, 142),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)


def main() -> None:
    """Flujo principal de captura de video y registro de datos bimanuales."""
    encabezados = generar_encabezados_csv()
    inicializar_archivo_csv(ARCHIVO_SALIDA_CSV, encabezados)

    detector_manos = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=CONFIANZA_MINIMA_DETECCION,
        min_tracking_confidence=CONFIANZA_MINIMA_SEGUIMIENTO
    )
    utilidad_dibujo = mp.solutions.drawing_utils

    captura = cv2.VideoCapture(0)
    indice_clase_actual = 0
    esta_grabando = False
    muestras_capturadas = 0

    print("=" * 60)
    print("CAPTURA DE DATASET LSP - MODO BIMANUAL (126 CARACTERÍSTICAS)")
    print("Controles:")
    print("  - Teclas [0 - 7] : Cambiar clase a registrar.")
    print("  - Barra Espacio  : Iniciar / Pausar grabación.")
    print("  - Tecla [Q]      : Salir y guardar.")
    print("=" * 60)

    try:
        while captura.isOpened():
            exito, fotograma = captura.read()
            if not exito:
                break

            fotograma = cv2.flip(fotograma, 1)
            fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
            resultados = detector_manos.process(fotograma_rgb)

            # Dibujar los puntos esqueléticos de las manos detectadas
            if resultados.multi_hand_landmarks:
                for marcas_mano in resultados.multi_hand_landmarks:
                    utilidad_dibujo.draw_landmarks(
                        fotograma, marcas_mano, mp.solutions.hands.HAND_CONNECTIONS
                    )

            vec_izq, vec_der, detectada_izq, detectada_der = extraer_vectores_manos(resultados)

            # Guardado si la grabación está activa y al menos una mano está visible
            if esta_grabando and (detectada_izq or detectada_der):
                vector_completo = vec_izq + vec_der + [indice_clase_actual, CLASES[indice_clase_actual]]
                pd.DataFrame([vector_completo]).to_csv(
                    ARCHIVO_SALIDA_CSV, mode="a", header=False, index=False
                )
                muestras_capturadas += 1
                time.sleep(INTERVALO_MUESTREO_SEGUNDOS)

            dibujar_interfaz(
                fotograma,
                indice_clase_actual,
                esta_grabando,
                muestras_capturadas,
                detectada_izq,
                detectada_der
            )

            cv2.imshow("Captura de Dataset LSP (Bimanual)", fotograma)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            elif tecla == 32:  # Barra espaciadora
                esta_grabando = not esta_grabando
            elif ord("0") <= tecla <= ord("7"):
                indice_clase_actual = tecla - ord("0")
                esta_grabando = False
                print(f"Seleccionada clase {indice_clase_actual}: {CLASES[indice_clase_actual]}")

    finally:
        captura.release()
        cv2.destroyAllWindows()
        detector_manos.close()


if __name__ == "__main__":
    main()
