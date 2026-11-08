"""Modulo de inferencia en tiempo real y banco de pruebas para el modelo LSP.

Implementa captura automatica de fotogramas por estabilidad de sena,
estampado visual de la clase detectada y despliegue del resultado en ventana dedicada.
"""

from datetime import datetime
import os
import time
from typing import Any
import cv2
import joblib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd

# ==============================================================================
# Constantes de Configuracion y Rutas
# ==============================================================================
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUTA_MODELO_SERIALIZADO = os.path.join(DIRECTORIO_RAIZ, "models", "modelo_lsp_ganador.joblib")
DIRECTORIO_EVIDENCIAS = os.path.join(DIRECTORIO_RAIZ, "reports", "evidencias")
RUTA_LOG_AUDITORIA = os.path.join(DIRECTORIO_RAIZ, "reports", "log_pruebas_en_vivo.csv")

# Parametros de disparo automatico de evidencias
UMBRAL_CONFIANZA_AUTO_PCT = 85.0
TIEMPO_ESTABILIDAD_SEGUNDOS = 0.75
ENFRIAMIENTO_MISMA_CLASE_SEG = 3.5
MAX_EVIDENCIAS_POR_CLASE = 3

# Parametros de vision y MediaPipe
CONFIANZA_MINIMA_DETECCION = 0.70
CONFIANZA_MINIMA_SEGUIMIENTO = 0.50
TOTAL_PUNTOS_POR_MANO = 21
DIMENSIONES_POR_PUNTO = 3
TOTAL_VALORES_POR_MANO = TOTAL_PUNTOS_POR_MANO * DIMENSIONES_POR_PUNTO

# Nombres de ventanas de visualizacion
VENTANA_CAMARA = "Inferencia en Vivo LSP - Camara Principal"
VENTANA_RESULTADO = "Resultado de Captura Automatica - Evidencia LSP"


def cargar_paquete_modelo(ruta_modelo: str) -> tuple[Any, list[str]]:
    """Carga el pipeline entrenado y el catalogo ordenado de clases desde joblib."""
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontro el modelo entrenado en: '{ruta_modelo}'")

    paquete = joblib.load(ruta_modelo)
    modelo = paquete["modelo"]
    clases = paquete["clases"]
    print(f"Modelo cargado: {paquete.get('nombre_modelo', 'SVM')} con {len(clases)} clases.")
    return modelo, clases


def inicializar_log_auditoria(ruta_log: str) -> None:
    """Genera la estructura inicial del archivo CSV de auditoria si no existe."""
    if not os.path.exists(ruta_log):
        df_base = pd.DataFrame(columns=[
            "timestamp", "clase_idx", "clase_nombre", "confianza_pct",
            "latencia_ms", "tipo_captura", "ruta_archivo"
        ])
        df_base.to_csv(ruta_log, index=False)


def registrar_evento_auditoria(
    ruta_log: str,
    clase_idx: int,
    clase_nombre: str,
    confianza: float,
    latencia_ms: float,
    tipo_captura: str,
    ruta_evidencia: str
) -> None:
    """Inserta un registro formal de evidencia en el archivo de auditoria."""
    registro = [{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "clase_idx": clase_idx,
        "clase_nombre": clase_nombre,
        "confianza_pct": round(confianza, 2),
        "latencia_ms": round(latencia_ms, 3),
        "tipo_captura": tipo_captura,
        "ruta_archivo": os.path.basename(ruta_evidencia)
    }]
    pd.DataFrame(registro).to_csv(ruta_log, mode="a", header=False, index=False)


def normalizar_puntos_mano(puntos_mano: Any) -> list[float]:
    """Calcula coordenadas 3D relativas al punto base (muneca) y normaliza su escala."""
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


def extraer_vectores_bimanuales(resultados: Any) -> tuple[list[float], list[float], bool, bool]:
    """Obtiene los vectores normalizados de 63 elementos para cada mano con zero-padding."""
    vector_izq = [0.0] * TOTAL_VALORES_POR_MANO
    vector_der = [0.0] * TOTAL_VALORES_POR_MANO
    detectada_izq = False
    detectada_der = False

    if not resultados.multi_hand_landmarks or not resultados.multi_handedness:
        return vector_izq, vector_der, detectada_izq, detectada_der

    for puntos, clasificacion in zip(resultados.multi_hand_landmarks, resultados.multi_handedness):
        etiqueta = clasificacion.classification[0].label
        vector_normalizado = normalizar_puntos_mano(puntos)

        if etiqueta == "Left":
            if not detectada_izq:
                vector_izq = vector_normalizado
                detectada_izq = True
            else:
                vector_der = vector_normalizado
                detectada_der = True
        elif etiqueta == "Right":
            if not detectada_der:
                vector_der = vector_normalizado
                detectada_der = True
            else:
                vector_izq = vector_normalizado
                detectada_izq = True

    return vector_izq, vector_der, detectada_izq, detectada_der


def calcular_caja_delimitadora_manos(
    resultados: Any,
    ancho_px: int,
    alto_px: int
) -> tuple[int, int, int, int] | None:
    """Calcula las coordenadas de la caja delimitadora envolvente de las manos detectadas."""
    if not resultados.multi_hand_landmarks:
        return None

    todos_x: list[int] = []
    todos_y: list[int] = []

    for marcas_mano in resultados.multi_hand_landmarks:
        for punto in marcas_mano.landmark:
            todos_x.append(int(punto.x * ancho_px))
            todos_y.append(int(punto.y * alto_px))

    if not todos_x or not todos_y:
        return None

    margen = 25
    x_min = max(0, min(todos_x) - margen)
    y_min = max(0, min(todos_y) - margen)
    x_max = min(ancho_px - 1, max(todos_x) + margen)
    y_max = min(alto_px - 1, max(todos_y) + margen)

    return x_min, y_min, x_max, y_max


def generar_fotograma_resultado(
    fotograma_base: np.ndarray,
    clase_idx: int,
    clase_nombre: str,
    confianza: float,
    latencia_ms: float,
    caja_manos: tuple[int, int, int, int] | None
) -> np.ndarray:
    """Genera una imagen de evidencia anotada indicando claramente la clase detectada."""
    resultado = fotograma_base.copy()
    alto_img, ancho_img, _ = resultado.shape
    marca_tiempo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Resaltar la region de las manos si esta disponible
    if caja_manos is not None:
        x1, y1, x2, y2 = caja_manos
        cv2.rectangle(resultado, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            resultado, f"Sena: {clase_nombre}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

    # Panel superior de encabezado tipo tarjeta de reporte
    alto_panel_sup = 85
    sub_img = resultado[0:alto_panel_sup, 0:ancho_img]
    rect_oscuro = np.zeros_like(sub_img, dtype=np.uint8)
    rect_oscuro[:] = (18, 18, 18)
    cv2.addWeighted(sub_img, 0.20, rect_oscuro, 0.80, 0, sub_img)
    resultado[0:alto_panel_sup, 0:ancho_img] = sub_img

    # Linea de acento verde
    cv2.line(resultado, (0, alto_panel_sup), (ancho_img, alto_panel_sup), (0, 220, 100), 3)

    # Texto principal: Clase detectada
    cv2.putText(
        resultado, f"CLASE DETECTADA: {clase_nombre} (ID: {clase_idx})",
        (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 120), 2
    )

    # Subtexto: Confianza, latencia y marca de tiempo
    texto_detalles = f"Confianza: {confianza:.1f}% | Latencia: {latencia_ms:.2f} ms | Fecha: {marca_tiempo}"
    cv2.putText(
        resultado, texto_detalles, (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1
    )

    # Insignia inferior de verificacion automatica
    alto_panel_inf = 35
    y_inf_inicio = alto_img - alto_panel_inf
    sub_inf = resultado[y_inf_inicio:alto_img, 0:ancho_img]
    rect_inf = np.zeros_like(sub_inf, dtype=np.uint8)
    rect_inf[:] = (10, 40, 10)
    cv2.addWeighted(sub_inf, 0.25, rect_inf, 0.75, 0, sub_inf)
    resultado[y_inf_inicio:alto_img, 0:ancho_img] = sub_inf

    cv2.putText(
        resultado, "EVIDENCIA CAPTURADA AUTOMATICAMENTE - RECONOCIMIENTO LSP / NAO",
        (20, alto_img - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1
    )

    return resultado


def guardar_evidencia_en_disco(
    fotograma_anotado: np.ndarray,
    clase_idx: int,
    clase_nombre: str,
    confianza: float,
    prefijo: str
) -> str:
    """Guarda en disco el fotograma de resultado con su nomenclatura estandar."""
    os.makedirs(DIRECTORIO_EVIDENCIAS, exist_ok=True)
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{prefijo}_evidencia_{clase_idx}_{clase_nombre}_conf{confianza:.1f}_{marca_tiempo}.png"
    ruta_guardado = os.path.join(DIRECTORIO_EVIDENCIAS, nombre_archivo)
    cv2.imwrite(ruta_guardado, fotograma_anotado)
    print(f"\n[EVIDENCIA GENERADA] {nombre_archivo}")
    return ruta_guardado


def dibujar_hud_principal(
    fotograma: np.ndarray,
    clase_nombre: str,
    confianza: float,
    latencia_ms: float,
    fps: float,
    progreso_auto: float,
    mensaje_notificacion: str,
    grabacion_activa: bool,
    mano_izq: bool,
    mano_der: bool,
    total_fotos_auto: int,
    miniatura_resultado: np.ndarray | None,
    clase_ultimo_resultado: str
) -> None:
    """Renderiza el panel HUD diagnostico, barra de progreso y miniatura del ultimo resultado."""
    alto_px, ancho_px, _ = fotograma.shape

    # Panel principal superior de telemetria
    cv2.rectangle(fotograma, (12, 12), (490, 185), (24, 24, 24), -1)
    cv2.rectangle(fotograma, (12, 12), (490, 185), (70, 70, 70), 1)

    if confianza >= UMBRAL_CONFIANZA_AUTO_PCT:
        color_estado = (0, 255, 100)
    elif confianza >= 55.0:
        color_estado = (0, 210, 255)
    else:
        color_estado = (160, 160, 160)

    # 1. Telemetria tecnica de velocidad
    cv2.putText(
        fotograma, f"LSP BANCO PRUEBAS | Latencia: {latencia_ms:.1f} ms | FPS: {fps:.0f}",
        (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1
    )

    # 2. Clase detectada en tiempo real
    texto_prediccion = f"Sena: {clase_nombre} ({confianza:.1f}%)" if (mano_izq or mano_der) else "Esperando senas..."
    cv2.putText(
        fotograma, texto_prediccion, (22, 76),
        cv2.FONT_HERSHEY_SIMPLEX, 0.82, color_estado, 2
    )

    # 3. Barra de estabilidad para disparo automatico
    cv2.putText(
        fotograma, "Captura automatica:", (22, 106),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1
    )
    ancho_barra = 210
    x_barra = 185
    y_barra = 96
    cv2.rectangle(fotograma, (x_barra, y_barra), (x_barra + ancho_barra, y_barra + 12), (60, 60, 60), 1)
    ancho_llenado = int(ancho_barra * min(progreso_auto, 1.0))
    if ancho_llenado > 0:
        color_barra = (0, 255, 0) if progreso_auto >= 1.0 else (0, 200, 255)
        cv2.rectangle(fotograma, (x_barra, y_barra), (x_barra + ancho_llenado, y_barra + 12), color_barra, -1)

    # 4. Estado bimanual y contador de evidencias
    texto_manos = f"Manos: Izq [{'SI' if mano_izq else 'NO'}] | Der [{'SI' if mano_der else 'NO'}]"
    cv2.putText(fotograma, texto_manos, (22, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 205, 255), 1)
    cv2.putText(
        fotograma, f"Fotos automaticas: {total_fotos_auto} capturas",
        (22, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 0), 1
    )

    # Indicador de grabacion de video clip
    if grabacion_activa:
        cv2.circle(fotograma, (ancho_px - 32, 32), 10, (0, 0, 255), -1)
        cv2.putText(fotograma, "REC", (ancho_px - 78, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Miniatura del ultimo resultado capturado (esquina inferior derecha)
    if miniatura_resultado is not None:
        ancho_mini = 180
        alto_mini = 120
        x_mini = ancho_px - ancho_mini - 16
        y_mini = alto_px - alto_mini - 20

        pequena = cv2.resize(miniatura_resultado, (ancho_mini, alto_mini))
        fotograma[y_mini:y_mini + alto_mini, x_mini:x_mini + ancho_mini] = pequena
        cv2.rectangle(fotograma, (x_mini, y_mini), (x_mini + ancho_mini, y_mini + alto_mini), (0, 255, 120), 2)

        cv2.rectangle(fotograma, (x_mini, y_mini - 20), (x_mini + ancho_mini, y_mini), (20, 20, 20), -1)
        cv2.putText(
            fotograma, f"Ultimo: {clase_ultimo_resultado}", (x_mini + 4, y_mini - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1
        )

    # Notificacion emergente
    if mensaje_notificacion:
        cv2.rectangle(fotograma, (12, alto_px - 50), (ancho_px - 12, alto_px - 14), (0, 140, 50), -1)
        cv2.putText(
            fotograma, mensaje_notificacion, (24, alto_px - 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )


def mostrar_galeria_capturas_final(evidencias: list[dict[str, Any]]) -> None:
    """Despliega una galeria grafica con todas las evidencias capturadas al finalizar la sesion."""
    if not evidencias:
        print("\nNo se registraron capturas de evidencias durante esta sesion.")
        return

    total = len(evidencias)
    columnas = min(3, total)
    filas = (total + columnas - 1) // columnas

    fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas))
    if hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title("Galeria Final de Evidencias LSP")

    if total == 1:
        axes = np.array([axes])
    axes = np.ravel(axes)

    for idx, ev in enumerate(evidencias):
        img_rgb = cv2.cvtColor(ev["fotograma"], cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(
            f"Captura {idx + 1}: {ev['clase_nombre']} ({ev['confianza']:.1f}%)\nLat: {ev['latencia_ms']:.1f} ms | {ev['tipo']}",
            fontsize=10, fontweight="bold", color="darkgreen"
        )
        axes[idx].axis("off")

    for sobrante in range(total, len(axes)):
        axes[sobrante].axis("off")

    plt.suptitle(
        f"Galeria Resumen de Evidencias Capturadas ({total} fotografias)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    print(f"\n[GALERIA FINAL] Desplegando mosaico con {total} evidencias capturadas...")
    plt.show()


def main() -> None:
    """Controla el flujo de prueba en vivo con captura automatica y despliegue de resultados."""
    print("=" * 72)
    print("BANCO DE PRUEBAS DE INFERENCIA EN VIVO Y CAPTURA AUTOMATICA (LSP)")
    print("=" * 72)

    modelo, nombres_clases = cargar_paquete_modelo(RUTA_MODELO_SERIALIZADO)
    inicializar_log_auditoria(RUTA_LOG_AUDITORIA)

    detector_manos = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=CONFIANZA_MINIMA_DETECCION,
        min_tracking_confidence=CONFIANZA_MINIMA_SEGUIMIENTO
    )
    utilidad_dibujo = mp.solutions.drawing_utils

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Error: No se pudo conectar a la camara web.")
        return

    # Contabilidad de capturas
    fotos_por_clase: dict[int, int] = {idx: 0 for idx in range(len(nombres_clases))}
    ultimo_disparo_por_clase: dict[int, float] = {idx: 0.0 for idx in range(len(nombres_clases))}
    evidencias_sesion: list[dict[str, Any]] = []

    # Estado de evaluacion continua
    clase_candidata_actual: int | None = None
    tiempo_inicio_postura: float | None = None

    # Estado de resultados y evidencia
    fotograma_ultimo_resultado: np.ndarray | None = None
    nombre_ultimo_resultado = "NINGUNO"

    # Grabacion auxiliar de video
    grabador_video: cv2.VideoWriter | None = None
    grabacion_activa = False

    # Mensajes al usuario
    texto_notificacion = ""
    tiempo_expira_notificacion = 0.0

    tiempo_anterior = time.time()
    latencia_ms = 0.0

    print("\nInstrucciones de operacion:")
    print("  - Realice una sena y sostengala 0.75 segundos con alta certeza.")
    print("  - El sistema capturara la evidencia automaticamente y mostrara el resultado.")
    print("  - [S] : Disparo manual forzado de evidencia con clase detectada.")
    print("  - [R] : Iniciar o detener grabacion de clip de video demostrativo.")
    print("  - [Q] : Cerrar el banco de pruebas.")
    print("=" * 72)

    try:
        while captura.isOpened():
            exito, fotograma = captura.read()
            if not exito:
                break

            tiempo_actual = time.time()
            fps = 1.0 / max(tiempo_actual - tiempo_anterior, 1e-4)
            tiempo_anterior = tiempo_actual

            fotograma = cv2.flip(fotograma, 1)
            alto_cam, ancho_cam, _ = fotograma.shape
            fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
            resultados = detector_manos.process(fotograma_rgb)

            # Dibujar esqueletos articulares en la camara principal
            if resultados.multi_hand_landmarks:
                for marcas_mano in resultados.multi_hand_landmarks:
                    utilidad_dibujo.draw_landmarks(
                        fotograma, marcas_mano, mp.solutions.hands.HAND_CONNECTIONS
                    )

            vec_izq, vec_der, detectada_izq, detectada_der = extraer_vectores_bimanuales(resultados)
            caja_manos = calcular_caja_delimitadora_manos(resultados, ancho_cam, alto_cam)

            clase_predicha_idx = 0
            nombre_predicho = "DESCONOCIDO"
            confianza_pct = 0.0
            progreso_auto = 0.0

            # Procesamiento de inferencia
            if detectada_izq or detectada_der:
                vector_entrada = np.array(vec_izq + vec_der, dtype=np.float32).reshape(1, -1)

                t_ini = time.perf_counter()
                probabilidades = modelo.predict_proba(vector_entrada)[0]
                t_fin = time.perf_counter()
                latencia_ms = (t_fin - t_ini) * 1000.0

                clase_predicha_idx = int(np.argmax(probabilidades))
                confianza_pct = float(probabilidades[clase_predicha_idx] * 100.0)
                nombre_predicho = nombres_clases[clase_predicha_idx]

                # Logica de disparo automatico por estabilidad
                if confianza_pct >= UMBRAL_CONFIANZA_AUTO_PCT:
                    if clase_candidata_actual == clase_predicha_idx and tiempo_inicio_postura is not None:
                        tiempo_sostenido = tiempo_actual - tiempo_inicio_postura
                        progreso_auto = tiempo_sostenido / TIEMPO_ESTABILIDAD_SEGUNDOS

                        if tiempo_sostenido >= TIEMPO_ESTABILIDAD_SEGUNDOS:
                            alcanzo_limite = fotos_por_clase[clase_predicha_idx] >= MAX_EVIDENCIAS_POR_CLASE
                            paso_enfriamiento = (
                                (tiempo_actual - ultimo_disparo_por_clase[clase_predicha_idx])
                                >= ENFRIAMIENTO_MISMA_CLASE_SEG
                            )

                            if not alcanzo_limite and paso_enfriamiento:
                                # 1. Generar fotograma de resultado con clase detectada indicada
                                img_resultado = generar_fotograma_resultado(
                                    fotograma, clase_predicha_idx, nombre_predicho,
                                    confianza_pct, latencia_ms, caja_manos
                                )

                                # 2. Guardar evidencia en disco y auditoria
                                ruta_foto = guardar_evidencia_en_disco(
                                    img_resultado, clase_predicha_idx, nombre_predicho,
                                    confianza_pct, "auto"
                                )
                                registrar_evento_auditoria(
                                    RUTA_LOG_AUDITORIA, clase_predicha_idx, nombre_predicho,
                                    confianza_pct, latencia_ms, "AUTOMATICA", ruta_foto
                                )

                                # 3. Actualizar estado y ventana de resultado
                                fotograma_ultimo_resultado = img_resultado
                                nombre_ultimo_resultado = nombre_predicho
                                fotos_por_clase[clase_predicha_idx] += 1
                                ultimo_disparo_por_clase[clase_predicha_idx] = tiempo_actual

                                evidencias_sesion.append({
                                    "fotograma": img_resultado.copy(),
                                    "clase_nombre": nombre_predicho,
                                    "confianza": confianza_pct,
                                    "latencia_ms": latencia_ms,
                                    "tipo": "AUTO"
                                })

                                texto_notificacion = f"Captura automatica completada: {nombre_predicho} ({confianza_pct:.1f}%)"
                                tiempo_expira_notificacion = tiempo_actual + 2.0
                                tiempo_inicio_postura = None
                    else:
                        clase_candidata_actual = clase_predicha_idx
                        tiempo_inicio_postura = tiempo_actual
                else:
                    clase_candidata_actual = None
                    tiempo_inicio_postura = None
            else:
                clase_candidata_actual = None
                tiempo_inicio_postura = None

            if tiempo_actual > tiempo_expira_notificacion:
                texto_notificacion = ""

            total_capturas = sum(fotos_por_clase.values())

            # Renderizar HUD en la camara en vivo
            dibujar_hud_principal(
                fotograma,
                nombre_predicho,
                confianza_pct,
                latencia_ms,
                fps,
                progreso_auto,
                texto_notificacion,
                grabacion_activa,
                detectada_izq,
                detectada_der,
                total_capturas,
                fotograma_ultimo_resultado,
                nombre_ultimo_resultado
            )

            if grabacion_activa and grabador_video is not None:
                grabador_video.write(fotograma)

            # Desplegar ventana principal de la camara
            cv2.imshow(VENTANA_CAMARA, fotograma)

            # Desplegar ventana de resultado cuando haya al menos una captura
            if fotograma_ultimo_resultado is not None:
                cv2.imshow(VENTANA_RESULTADO, fotograma_ultimo_resultado)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            elif tecla == ord("s"):
                # Disparo manual forzado con indicacion de clase
                img_manual = generar_fotograma_resultado(
                    fotograma, clase_predicha_idx, nombre_predicho,
                    confianza_pct, latencia_ms, caja_manos
                )
                ruta_man = guardar_evidencia_en_disco(
                    img_manual, clase_predicha_idx, nombre_predicho,
                    confianza_pct, "manual"
                )
                registrar_evento_auditoria(
                    RUTA_LOG_AUDITORIA, clase_predicha_idx, nombre_predicho,
                    confianza_pct, latencia_ms, "MANUAL", ruta_man
                )
                fotograma_ultimo_resultado = img_manual
                nombre_ultimo_resultado = nombre_predicho

                evidencias_sesion.append({
                    "fotograma": img_manual.copy(),
                    "clase_nombre": nombre_predicho,
                    "confianza": confianza_pct,
                    "latencia_ms": latencia_ms,
                    "tipo": "MANUAL"
                })

                texto_notificacion = f"Captura manual guardada: {nombre_predicho} ({confianza_pct:.1f}%)"
                tiempo_expira_notificacion = time.time() + 2.0
            elif tecla == ord("r"):
                grabacion_activa = not grabacion_activa
                if grabacion_activa:
                    os.makedirs(DIRECTORIO_EVIDENCIAS, exist_ok=True)
                    marca_clip = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ruta_clip = os.path.join(DIRECTORIO_EVIDENCIAS, f"clip_demostrativo_{marca_clip}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    grabador_video = cv2.VideoWriter(ruta_clip, fourcc, 20.0, (ancho_cam, alto_cam))
                    print(f"\n[GRABACION INICIADA] {ruta_clip}")
                    texto_notificacion = "Grabando clip de video MP4..."
                    tiempo_expira_notificacion = time.time() + 2.0
                else:
                    if grabador_video is not None:
                        grabador_video.release()
                        grabador_video = None
                    print("\n[GRABACION DETENIDA]")
                    texto_notificacion = "Clip de video guardado con exito."
                    tiempo_expira_notificacion = time.time() + 2.0

    finally:
        captura.release()
        if grabador_video is not None:
            grabador_video.release()
        cv2.destroyAllWindows()
        detector_manos.close()
        print("\nSesion de pruebas finalizada.")
        print(f"Total de evidencias capturadas: {len(evidencias_sesion)}")
        print(f"Directorio de almacenamiento: '{DIRECTORIO_EVIDENCIAS}'")
        print(f"Registro de auditoria: '{RUTA_LOG_AUDITORIA}'")

        # Mostrar galeria completa de resultados al finalizar la ejecucion
        mostrar_galeria_capturas_final(evidencias_sesion)


if __name__ == "__main__":
    main()
