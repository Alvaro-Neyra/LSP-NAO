"""Aplicacion de Inferencia en Tiempo Real y Maquina de Estados para el Robot NAO.

Captura fotogramas de la webcam, extrae caracteristicas articulares con MediaPipe Hands,
clasifica senas de Lengua de Senas Peruana (LSP) con SVM, filtra por consistencia
temporal (debounce FSM), arma oraciones contextuales y las transmite por socket TCP
al servidor de control del robot NAO (Python 2.7 / NAOqi).
"""

from datetime import datetime
import json
import os
import socket
import sys
import time
from typing import Any
import cv2
import joblib
import mediapipe as mp
import numpy as np

# ==============================================================================
# Rutas y Constantes del Sistema
# ==============================================================================
def obtener_directorio_raiz() -> str:
    """Calcula la ruta raiz del proyecto buscando la carpeta 'models'."""
    ruta_actual = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.exists(os.path.join(ruta_actual, "models")):
            return ruta_actual
        padre = os.path.dirname(ruta_actual)
        if padre == ruta_actual:
            break
        ruta_actual = padre
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DIRECTORIO_RAIZ = obtener_directorio_raiz()
RUTA_MODELO_SERIALIZADO = os.path.join(DIRECTORIO_RAIZ, "models", "modelo_lsp_ganador.joblib")

# Parametros de comunicacion Socket TCP con el puente NAOqi
HOST_SOCKET_DEFECTO = "127.0.0.1"
PUERTO_SOCKET_DEFECTO = 50005
TIMEOUT_SOCKET_SEG = 0.5

# Parametros del Filtro Temporal y FSM
FRAMES_CONSISTENCIA_REQUERIDOS = 14  # Frames continuos para aceptar un token (~0.5 - 0.6s)
UMBRAL_CONFIANZA_MINIMA = 84.0        # Porcentaje minimo de certeza del clasificador
ENFRIAMIENTO_MISMO_TOKEN_SEG = 2.0   # Evita duplicar el mismo token si se mantiene la postura

# Parametros de MediaPipe
CONFIANZA_DETECCION_MANOS = 0.70
CONFIANZA_SEGUIMIENTO_MANOS = 0.50
TOTAL_PUNTOS_POR_MANO = 21
DIMENSIONES_POR_PUNTO = 3
TOTAL_VALORES_POR_MANO = TOTAL_PUNTOS_POR_MANO * DIMENSIONES_POR_PUNTO


class ClienteSocketNao:
    """Gestiona la conexion TCP/IP no bloqueante con el servidor puente de NAOqi."""

    def __init__(self, host: str, puerto: int) -> None:
        self.host = host
        self.puerto = puerto
        self.sock: socket.socket | None = None
        self.esta_conectado = False
        self.ultimo_intento_reconexion = 0.0
        self.conectar()

    def conectar(self) -> bool:
        """Intenta establecer la comunicacion con el servidor en Python 2.7."""
        ahora = time.time()
        if ahora - self.ultimo_intento_reconexion < 3.0:
            return False
        self.ultimo_intento_reconexion = ahora

        try:
            self.desconectar()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(TIMEOUT_SOCKET_SEG)
            self.sock.connect((self.host, self.puerto))
            self.esta_conectado = True
            print(f"[SOCKET CONECTADO] Conectado al puente NAOqi en {self.host}:{self.puerto}")
            return True
        except (socket.error, socket.timeout):
            self.esta_conectado = False
            return False

    def enviar_evento(self, carga_datos: dict[str, Any]) -> bool:
        """Serializa en JSON y transmite un comando al robot NAO."""
        if not self.esta_conectado or self.sock is None:
            # Intento de reconexion transparente
            if not self.conectar():
                return False

        try:
            mensaje = json.dumps(carga_datos) + "\n"
            self.sock.sendall(mensaje.encode("utf-8"))
            return True
        except (socket.error, socket.timeout) as error:
            print(f"[AVISO SOCKET] Falla al enviar datos a NAO: {error}")
            self.esta_conectado = False
            self.desconectar()
            return False

    def desconectar(self) -> None:
        """Cierra el descriptor de socket de forma segura."""
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.esta_conectado = False


def normalizar_puntos_mano(puntos_mano: Any) -> list[float]:
    """Traslada las coordenadas respecto a la muneca y normaliza por distancia maxima."""
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


def extraer_descriptores_bimanuales(resultados: Any) -> tuple[list[float], list[float], bool, bool]:
    """Construye los vectores normalizados de 63 elementos para cada mano con zero-padding."""
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


def componer_oracion_gramatical(tokens: list[str]) -> str:
    """Transforma una secuencia de tokens LSP en una oracion gramatical fluida en espanol."""
    if not tokens:
        return ""

    patron = " ".join(tokens)

    # Patrones de 3 tokens frecuentes
    if patron == "YO NECESITAR AGUA":
        return "Yo necesito agua."
    elif patron == "YO NECESITAR AYUDA":
        return "Yo necesito ayuda urgente."
    elif patron == "YO NECESITAR MEDICINA":
        return "Yo necesito mi medicina."
    elif patron == "YO QUERER AGUA":
        return "Yo quiero tomar agua."
    elif patron == "YO QUERER MEDICINA":
        return "Yo quiero mi medicina."
    elif patron == "YO QUERER AYUDA":
        return "Yo quiero que me ayudes."
    elif patron == "TU NECESITAR AGUA":
        return "Tu necesitas agua."
    elif patron == "TU NECESITAR AYUDA":
        return "Tu necesitas ayuda."
    elif patron == "TU NECESITAR MEDICINA":
        return "Tu necesitas medicina."
    elif patron == "TU QUERER AGUA":
        return "Tu quieres agua."
    elif patron == "TU QUERER AYUDA":
        return "Tu quieres ayuda."
    elif patron == "TU QUERER MEDICINA":
        return "Tu quieres medicina."

    # Patrones de 2 tokens
    elif patron == "NECESITAR AGUA":
        return "Necesito agua, por favor."
    elif patron == "NECESITAR AYUDA":
        return "Necesito ayuda urgente."
    elif patron == "NECESITAR MEDICINA":
        return "Necesito medicina, por favor."
    elif patron == "QUERER AGUA":
        return "Quiero tomar agua."
    elif patron == "QUERER MEDICINA":
        return "Quiero mi medicina."
    elif patron == "QUERER AYUDA":
        return "Requiero asistencia."
    elif patron == "YO AGUA":
        return "Yo necesito agua."
    elif patron == "YO AYUDA":
        return "Yo solicito ayuda."
    elif patron == "YO MEDICINA":
        return "Yo necesito medicina."

    # Patrones de 1 solo token
    elif patron == "AYUDA":
        return "Por favor, necesito ayuda inmediata."
    elif patron == "AGUA":
        return "Por favor, solicito un vaso de agua."
    elif patron == "MEDICINA":
        return "Por favor, necesito tomar mi medicina."
    elif patron == "NECESITAR":
        return "Tengo una necesidad que comunicar."
    elif patron == "QUERER":
        return "Deseo comunicar una solicitud."
    elif patron == "YO":
        return "Me refiero a mi persona."
    elif patron == "TU":
        return "Me dirijo hacia ti."

    # Ensamblador heuristico general con reglas de concordancia
    elementos = []
    sujeto = None

    for tok in tokens:
        if tok == "YO":
            sujeto = "YO"
            elementos.append("Yo")
        elif tok == "TU":
            sujeto = "Tu"
            elementos.append("Tu")
        elif tok == "NECESITAR":
            if sujeto == "YO":
                elementos.append("necesito")
            elif sujeto == "Tu":
                elementos.append("necesitas")
            else:
                elementos.append("necesito")
        elif tok == "QUERER":
            if sujeto == "YO":
                elementos.append("quiero")
            elif sujeto == "Tu":
                elementos.append("quieres")
            else:
                elementos.append("quiero")
        elif tok == "AGUA":
            elementos.append("agua")
        elif tok == "AYUDA":
            elementos.append("ayuda")
        elif tok == "MEDICINA":
            elementos.append("medicina")
        else:
            elementos.append(tok.lower())

    frase = " ".join(elementos).strip()
    if frase and not frase.endswith("."):
        frase += "."
    return frase[0].upper() + frase[1:] if frase else ""


def dibujar_panel_hud(
    fotograma: np.ndarray,
    sena_actual: str,
    confianza: float,
    progreso_consistencia: float,
    bufer_tokens: list[str],
    oracion_preview: str,
    estado_socket: bool,
    mensaje_notificacion: str,
    fps: float
) -> None:
    """Renderiza la interfaz semi-transparente de monitoreo y telemetria sobre el video."""
    alto, ancho, _ = fotograma.shape

    # 1. Panel Superior Izquierdo: Telemetria de Inferencia y Consistencia Temporal
    x1, y1, x2, y2 = 15, 15, 490, 195
    sub_roi = fotograma[y1:y2, x1:x2]
    overlay_oscuro = np.zeros_like(sub_roi, dtype=np.uint8)
    overlay_oscuro[:] = (20, 20, 20)
    cv2.addWeighted(sub_roi, 0.25, overlay_oscuro, 0.75, 0, sub_roi)
    fotograma[y1:y2, x1:x2] = sub_roi
    cv2.rectangle(fotograma, (x1, y1), (x2, y2), (70, 70, 70), 1)

    # Estado de color segun confianza
    if confianza >= UMBRAL_CONFIANZA_MINIMA:
        color_sena = (0, 255, 120)
    elif confianza >= 50.0:
        color_sena = (0, 215, 255)
    else:
        color_sena = (150, 150, 150)

    # Linea de estado y FPS
    cv2.putText(
        fotograma, f"LSP -> ROBOT NAO | FPS: {fps:.0f}",
        (x1 + 12, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
    )

    # Sena detectada
    texto_deteccion = f"Sena: {sena_actual} ({confianza:.1f}%)" if sena_actual != "REPOSO" else "Esperando senas..."
    cv2.putText(
        fotograma, texto_deteccion,
        (x1 + 12, y1 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color_sena, 2
    )

    # Barra de progreso temporal (Debounce)
    cv2.putText(
        fotograma, "Consistencia temporal:",
        (x1 + 12, y1 + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1
    )
    ancho_b = 200
    cv2.rectangle(fotograma, (x1 + 175, y1 + 80), (x1 + 175 + ancho_b, y1 + 95), (60, 60, 60), 1)
    len_barra = int(ancho_b * min(progreso_consistencia, 1.0))
    if len_barra > 0:
        col_b = (0, 255, 0) if progreso_consistencia >= 1.0 else (0, 200, 255)
        cv2.rectangle(fotograma, (x1 + 175, y1 + 80), (x1 + 175 + len_barra, y1 + 95), col_b, -1)

    # Estado del Socket con NAO
    color_st = (0, 255, 120) if estado_socket else (0, 0, 255)
    txt_st = "CONECTADO A NAOQI" if estado_socket else "DESCONECTADO (REINTENTANDO)"
    cv2.circle(fotograma, (x1 + 22, y1 + 125), 6, color_st, -1)
    cv2.putText(
        fotograma, f"Puente NAO: {txt_st}",
        (x1 + 36, y1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_st, 1
    )

    # Controles de teclado
    cv2.putText(
        fotograma, "[C]: Borrar Bufer | [S]: Forzar Envio | [Q]: Salir",
        (x1 + 12, y1 + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1
    )

    # 2. Panel Inferior: Bufer de Frase y Oracion Sintetizada
    y_inf_ini = alto - 105
    sub_inf = fotograma[y_inf_ini:alto - 15, 15:ancho - 15]
    overlay_inf = np.zeros_like(sub_inf, dtype=np.uint8)
    overlay_inf[:] = (18, 18, 18)
    cv2.addWeighted(sub_inf, 0.20, overlay_inf, 0.80, 0, sub_inf)
    fotograma[y_inf_ini:alto - 15, 15:ancho - 15] = sub_inf
    cv2.rectangle(fotograma, (15, y_inf_ini), (ancho - 15, alto - 15), (0, 200, 120), 2)

    cadena_bufer = " -> ".join(bufer_tokens) if bufer_tokens else "[Vacio - Realice una sena para agregar]"
    cv2.putText(
        fotograma, f"BUFER TOKENS: {cadena_bufer}",
        (25, y_inf_ini + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2
    )

    cv2.putText(
        fotograma, f"ORACION NAO: \"{oracion_preview}\"",
        (25, y_inf_ini + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )

    # Notificacion temporal central si existe
    if mensaje_notificacion:
        cv2.rectangle(fotograma, (ancho // 4, 20), (3 * ancho // 4, 65), (0, 140, 50), -1)
        cv2.putText(
            fotograma, mensaje_notificacion,
            (ancho // 4 + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )


def main() -> None:
    """Ejecuta el pipeline completo de inferencia, FSM y transmision hacia el robot NAO."""
    print("=" * 72)
    print(" APLICACION DE INFERENCIA LSP Y TRADUCCION CON ROBOT NAO")
    print("=" * 72)

    if not os.path.exists(RUTA_MODELO_SERIALIZADO):
        print(f"Error critico: No se encontro el modelo en '{RUTA_MODELO_SERIALIZADO}'.")
        return

    paquete = joblib.load(RUTA_MODELO_SERIALIZADO)
    modelo = paquete["modelo"]
    nombres_clases = paquete["clases"]
    print(f"Modelo cargado: {paquete.get('nombre_modelo', 'SVM')} con {len(nombres_clases)} clases.")

    # Conectar cliente socket con el puente Python 2.7
    cliente_nao = ClienteSocketNao(HOST_SOCKET_DEFECTO, PUERTO_SOCKET_DEFECTO)

    detector_manos = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=CONFIANZA_DETECCION_MANOS,
        min_tracking_confidence=CONFIANZA_SEGUIMIENTO_MANOS
    )
    utilidad_dibujo = mp.solutions.drawing_utils

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Error: No se pudo abrir la camara web.")
        return

    # Variables de estado de la Maquina de Estados Finita (FSM)
    bufer_tokens: list[str] = []
    conteo_frames_consecutivos = 0
    candidato_actual: str | None = None
    ultimo_token_aceptado: str | None = None
    tiempo_ultimo_token = 0.0

    notificacion_texto = ""
    tiempo_expira_notificacion = 0.0
    tiempo_anterior = time.time()

    print("\nInstrucciones del Sistema Asistencial:")
    print("  1. Sostenga una sena para registrar el token (se emite feedback verde en NAO).")
    print("  2. Acumule los conceptos deseados (ej: YO -> NECESITAR -> AGUA).")
    print("  3. Realice la sena 'ENVIAR' para que NAO vocalice la oracion y ejecute el gesto.")
    print("  4. Presione [C] para borrar el bufer, [S] para forzar envio y [Q] para salir.\n")

    try:
        while captura.isOpened():
            exito, fotograma = captura.read()
            if not exito:
                break

            tiempo_actual = time.time()
            fps = 1.0 / max(tiempo_actual - tiempo_anterior, 1e-4)
            tiempo_anterior = tiempo_actual

            fotograma = cv2.flip(fotograma, 1)
            fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
            resultados = detector_manos.process(fotograma_rgb)

            # Dibujar marcas articulares de las manos
            if resultados.multi_hand_landmarks:
                for marcas in resultados.multi_hand_landmarks:
                    utilidad_dibujo.draw_landmarks(
                        fotograma, marcas, mp.solutions.hands.HAND_CONNECTIONS
                    )

            vec_izq, vec_der, det_izq, det_der = extraer_descriptores_bimanuales(resultados)

            sena_detectada = "REPOSO"
            confianza_actual = 0.0
            progreso_debounce = 0.0

            # Inferencia cuando al menos una mano esta visible
            if det_izq or det_der:
                vector_entrada = np.array(vec_izq + vec_der, dtype=np.float32).reshape(1, -1)
                probabilidades = modelo.predict_proba(vector_entrada)[0]
                idx_predicho = int(np.argmax(probabilidades))
                confianza_actual = float(probabilidades[idx_predicho] * 100.0)
                sena_detectada = nombres_clases[idx_predicho]

                # ==============================================================
                # Logica de la Maquina de Estados (Debounce y Bufer)
                # ==============================================================
                if confianza_actual >= UMBRAL_CONFIANZA_MINIMA:
                    if sena_detectada == candidato_actual:
                        conteo_frames_consecutivos += 1
                        progreso_debounce = conteo_frames_consecutivos / float(FRAMES_CONSISTENCIA_REQUERIDOS)

                        # Se supero el umbral de estabilidad temporal
                        if conteo_frames_consecutivos >= FRAMES_CONSISTENCIA_REQUERIDOS:
                            paso_enfriamiento = (
                                (tiempo_actual - tiempo_ultimo_token) >= ENFRIAMIENTO_MISMO_TOKEN_SEG
                                or sena_detectada != ultimo_token_aceptado
                            )

                            if paso_enfriamiento:
                                if sena_detectada == "ENVIAR":
                                    # DISPARADOR DE ENVIO FINAL
                                    if bufer_tokens:
                                        oracion_final = componer_oracion_gramatical(bufer_tokens)
                                        print(f"\n[DISPARO ENVIAR] Transmitiendo oracion: \"{oracion_final}\"")

                                        cliente_nao.enviar_evento({
                                            "tipo": "ORACION",
                                            "frase": oracion_final,
                                            "tokens": bufer_tokens
                                        })

                                        notificacion_texto = f"Enviado a NAO: \"{oracion_final}\""
                                        tiempo_expira_notificacion = tiempo_actual + 3.0
                                        bufer_tokens = []
                                    else:
                                        notificacion_texto = "Bufer vacio. Ingrese senas antes de enviar."
                                        tiempo_expira_notificacion = tiempo_actual + 2.0
                                else:
                                    # REGISTRO DE NUEVO TOKEN
                                    bufer_tokens.append(sena_detectada)
                                    print(f"[TOKEN ANADIDO] +{sena_detectada} | Bufer actual: {bufer_tokens}")

                                    # Notificar feedback verde instantaneo a NAO
                                    cliente_nao.enviar_evento({
                                        "tipo": "TOKEN",
                                        "token": sena_detectada
                                    })

                                    notificacion_texto = f"Token aceptado: {sena_detectada}"
                                    tiempo_expira_notificacion = tiempo_actual + 1.5

                                ultimo_token_aceptado = sena_detectada
                                tiempo_ultimo_token = tiempo_actual
                                conteo_frames_consecutivos = 0
                                candidato_actual = None
                    else:
                        candidato_actual = sena_detectada
                        conteo_frames_consecutivos = 1
                        progreso_debounce = 1.0 / FRAMES_CONSISTENCIA_REQUERIDOS
                else:
                    candidato_actual = None
                    conteo_frames_consecutivos = 0
                    progreso_debounce = 0.0
            else:
                candidato_actual = None
                conteo_frames_consecutivos = 0
                progreso_debounce = 0.0

            if tiempo_actual > tiempo_expira_notificacion:
                notificacion_texto = ""

            oracion_preview = componer_oracion_gramatical(bufer_tokens)

            # Dibujar la interfaz grafica HUD
            dibujar_panel_hud(
                fotograma,
                sena_detectada,
                confianza_actual,
                progreso_debounce,
                bufer_tokens,
                oracion_preview,
                cliente_nao.esta_conectado,
                notificacion_texto,
                fps
            )

            cv2.imshow("Traductor de Senas LSP a Voz y Movimiento - Robot NAO", fotograma)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            elif tecla in (ord("c"), ord("C"), 8):  # Tecla C o Backspace
                bufer_tokens = []
                cliente_nao.enviar_evento({"tipo": "RESET"})
                notificacion_texto = "Bufer reiniciado."
                tiempo_expira_notificacion = time.time() + 1.5
                print("[RESET] Bufer vaciado por comando de teclado.")
            elif tecla in (ord("s"), ord("S")):  # Forzar envio
                if bufer_tokens:
                    oracion_forzada = componer_oracion_gramatical(bufer_tokens)
                    print(f"\n[ENVIO FORZADO] Transmitiendo: \"{oracion_forzada}\"")
                    cliente_nao.enviar_evento({
                        "tipo": "ORACION",
                        "frase": oracion_forzada,
                        "tokens": bufer_tokens
                    })
                    notificacion_texto = f"Forzado a NAO: \"{oracion_forzada}\""
                    tiempo_expira_notificacion = time.time() + 3.0
                    bufer_tokens = []
                else:
                    notificacion_texto = "Bufer vacio, nada que enviar."
                    tiempo_expira_notificacion = time.time() + 1.5

    finally:
        captura.release()
        cv2.destroyAllWindows()
        detector_manos.close()
        cliente_nao.desconectar()
        print("\nAplicacion de inferencia finalizada con exito.")


if __name__ == "__main__":
    main()
