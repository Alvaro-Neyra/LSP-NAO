# -*- coding: utf-8 -*-
"""Servidor Puente Socket TCP para el Robot Humanoide NAO (Python 2.7).

Recibe comandos de control, tokens aceptados y oraciones armadas desde el
modulo de inferencia en Python 3, coordinando la sintesis de voz (ALTextToSpeech),
gestos cinesteticos (ALMotion, ALRobotPosture) y senalizacion visual (ALLeds).
"""

import argparse
import json
import socket
import struct
import sys
import threading
import time

# Intentar importar el SDK oficial de Aldebaran Robotics (NAOqi)
SIMULACION_MOCK = False
try:
    from naoqi import ALProxy
except ImportError:
    SIMULACION_MOCK = True
    print("[AVISO] SDK 'naoqi' no detectado en el entorno local.")
    print("[MODO] Activando controlador simulado Mock de prueba en consola.\n")

# Constantes del Servidor y Parametros por Defecto
HOST_SOCKET_DEFECTO = "127.0.0.1"
PUERTO_SOCKET_DEFECTO = 50005
PUERTO_VIDEO_DEFECTO = 50006
TAMANO_BUFFER = 4096

NAO_IP_DEFECTO = "127.0.0.1"
NAO_PUERTO_DEFECTO = 56432
POSTURA_BASE = "StandInit"
VELOCIDAD_POSTURA = 0.6

# Parametros de Camara NAOqi (ALVideoDevice)
CAMARA_SUPERIOR_ID = 0
CAMARA_RESOLUCION_VGA = 2      # 640 x 480
CAMARA_RESOLUCION_QVGA = 1     # 320 x 240 (Optimo para Wi-Fi sin latencia)
ESPACIO_COLOR_RGB = 11         # kRGBColorSpace
ESPACIO_COLOR_BGR = 13         # kBGRColorSpace
FPS_CAMARA_DEFECTO = 15        # Tasa de cuadros equilibrada para evitar congestion en red


class ControladorRobotNao(object):
    """Encapsula los proxies de NAOqi para coordinar voz, postura, movimiento, luces y video."""

    def __init__(self, ip_nao, puerto_nao, usar_mock=False,
                 habilitar_camara=True, cam_id=CAMARA_SUPERIOR_ID,
                 cam_res=CAMARA_RESOLUCION_QVGA, cam_fps=FPS_CAMARA_DEFECTO):
        self.ip_nao = ip_nao
        self.puerto_nao = int(puerto_nao)
        self.usar_mock = usar_mock
        self.habilitar_camara = habilitar_camara
        self.cam_id = cam_id
        self.cam_res = cam_res
        self.cam_fps = cam_fps
        self.tts = None
        self.postura = None
        self.motion = None
        self.leds = None
        self.cam_proxy = None
        self.id_suscriptor_camara = None
        self.espacio_color_actual = ESPACIO_COLOR_BGR
        self.inicializar_proxies()

    def inicializar_proxies(self):
        """Inicializa proxies de comunicacion con el middleware NAOqi."""
        if self.usar_mock:
            print("[NAO MOCK] Proxies simulados inicializados con exito.")
            return

        try:
            print("[CONECTANDO] Conectando a NAOqi en {}:{}...".format(self.ip_nao, self.puerto_nao))
            self.tts = ALProxy("ALTextToSpeech", self.ip_nao, self.puerto_nao)
            self.postura = ALProxy("ALRobotPosture", self.ip_nao, self.puerto_nao)
            self.motion = ALProxy("ALMotion", self.ip_nao, self.puerto_nao)
            self.leds = ALProxy("ALLeds", self.ip_nao, self.puerto_nao)

            # Configuracion de idioma en espanol y volumen
            try:
                self.tts.setLanguage("Spanish")
            except Exception as e_lang:
                print("[AVISO] No se pudo configurar idioma Spanish en TTS: {}".format(e_lang))

            # Habilitar rigidez y enviar postura base segura
            self.motion.wakeUp()
            self.postura.goToPosture(POSTURA_BASE, VELOCIDAD_POSTURA)
            self.establecer_leds_base()
            print("[NAO CONECTADO] Robot listo en postura '{}' con rigidez activa.".format(POSTURA_BASE))

            # Inicializar modulo de vision (ALVideoDevice)
            if self.habilitar_camara:
                self._inicializar_camara()

        except Exception as error:
            print("[ERROR] Falla al conectar con los proxies de NAOqi: {}".format(error))
            print("[FALLBACK] Conmutando a modo simulado Mock para mantener el servidor activo.")
            self.usar_mock = True

    def _inicializar_camara(self):
        """Suscribe un cliente de transmision de video en ALVideoDevice."""
        try:
            self.cam_proxy = ALProxy("ALVideoDevice", self.ip_nao, self.puerto_nao)
            # Intentar primero en formato BGR nativo para OpenCV
            try:
                self.id_suscriptor_camara = self.cam_proxy.subscribeCamera(
                    "LSP_Stream_Cam", self.cam_id, self.cam_res, ESPACIO_COLOR_BGR, self.cam_fps
                )
                self.espacio_color_actual = ESPACIO_COLOR_BGR
                print("[CAMARA CONECTADA] Suscrito a ALVideoDevice en formato BGR (ID: {})".format(self.id_suscriptor_camara))
            except Exception as err_bgr:
                print("[AVISO CAMARA] Modo BGR no admitido ({}), reintentando RGB...".format(err_bgr))
                self.id_suscriptor_camara = self.cam_proxy.subscribeCamera(
                    "LSP_Stream_Cam", self.cam_id, self.cam_res, ESPACIO_COLOR_RGB, self.cam_fps
                )
                self.espacio_color_actual = ESPACIO_COLOR_RGB
                print("[CAMARA CONECTADA] Suscrito a ALVideoDevice en formato RGB (ID: {})".format(self.id_suscriptor_camara))
        except Exception as err_cam:
            print("[ERROR CAMARA] No se pudo suscribir a ALVideoDevice: {}".format(err_cam))
            self.cam_proxy = None
            self.id_suscriptor_camara = None

    def obtener_fotograma_actual(self):
        """Retorna (ancho, alto, canales, espacio_color, bytes_crudos) del fotograma actual."""
        if self.usar_mock or self.cam_proxy is None or self.id_suscriptor_camara is None:
            # Fotograma sintetico mock para pruebas sin robot fisico (640x480)
            ancho, alto, canales = 640, 480, 3
            linea_mock = ("\x1e\x1e\x1e" * 320) + ("\x20\xa0\x20" * 320)
            datos_mock = linea_mock * alto
            return ancho, alto, canales, ESPACIO_COLOR_BGR, datos_mock

        try:
            resultado = self.cam_proxy.getImageRemote(self.id_suscriptor_camara)
            if resultado is None:
                return None
            ancho = int(resultado[0])
            alto = int(resultado[1])
            canales = int(resultado[2])
            espacio_color = int(resultado[3])
            bytes_imagen = resultado[6]

            # Liberar buffer de imagen en NAOqi para evitar retardo en cola
            try:
                self.cam_proxy.releaseImage(self.id_suscriptor_camara)
            except Exception:
                pass

            return ancho, alto, canales, espacio_color, bytes_imagen
        except Exception as error:
            print("[ERROR CAPTURA VIDEO] {}".format(error))
            return None

    def establecer_leds_base(self):
        """Establece los LEDs faciales en color blanco tenue de reposo."""
        if self.usar_mock:
            return
        try:
            self.leds.fadeRGB("FaceLeds", 0.7, 0.7, 0.7, 0.2)
        except Exception as e:
            print("[ERROR LEDS] {}".format(e))

    def notificar_token_aceptado(self, token_nombre):
        """Genera retroalimentacion visual verde de 0.3s confirmando un token aceptado."""
        print("[TOKEN RECIBIDO] {}".format(token_nombre))
        if self.usar_mock:
            print("  -> [MOCK LEDS] Ojos VERDES por 0.3s (Token: {})".format(token_nombre))
            return

        try:
            # Color verde brillante en los ojos
            self.leds.fadeRGB("FaceLeds", 0.0, 1.0, 0.0, 0.05)
            time.sleep(0.3)
            self.establecer_leds_base()
        except Exception as error:
            print("[ERROR FEEDBACK TOKEN] {}".format(error))

    def ejecutar_animacion_asentimiento(self):
        """Ejecuta una animacion suave y cinesteticamente segura de asentimiento con brazos."""
        if self.usar_mock:
            print("  -> [MOCK MOTION] Animacion de confirmacion cinestetica ejecutada.")
            return

        try:
            # Asentimiento de cabeza y extension ligera de antebrazos
            nombres_articulaciones = ["HeadPitch", "RShoulderPitch", "LShoulderPitch", "RElbowRoll", "LElbowRoll"]
            angulos_objetivo = [0.15, 1.2, 1.2, 0.6, -0.6]
            tiempos = [0.8, 0.8, 0.8, 0.8, 0.8]
            self.motion.angleInterpolation(nombres_articulaciones, angulos_objetivo, tiempos, True)

            # Breve pausa y retorno a postura base
            time.sleep(0.3)
            self.postura.goToPosture(POSTURA_BASE, VELOCIDAD_POSTURA)
        except Exception as error:
            print("[ERROR ANIMACION] {}".format(error))
            try:
                self.postura.goToPosture(POSTURA_BASE, VELOCIDAD_POSTURA)
            except Exception:
                pass

    def verbalizar_oracion_completa(self, texto_oracion, tokens):
        """Ejecuta la orden final: iluminacion azul, sintesis de voz y gesto coordinado."""
        print("\n" + "=" * 60)
        print("[ORDEN DE EJECUCION FINAL - ORACION ARMADA]")
        print("  Tokens de entrada : {}".format(tokens))
        print("  Frase a vocalizar : \"{}\"".format(texto_oracion))
        print("=" * 60)

        # 1. Indicador visual: LEDs en azul cian de comunicacion activa
        if not self.usar_mock:
            try:
                self.leds.fadeRGB("FaceLeds", 0.0, 0.6, 1.0, 0.15)
            except Exception as e:
                print("[ERROR LEDS] {}".format(e))
        else:
            print("  -> [MOCK LEDS] Ojos AZUL CIAN activo.")

        # 2. Vocalizacion con ALTextToSpeech
        if not self.usar_mock and self.tts is not None:
            try:
                # Asegurar codificacion utf-8 para texto en espanol
                cadena_utf8 = texto_oracion.encode("utf-8") if isinstance(texto_oracion, unicode) else texto_oracion
                self.tts.say(cadena_utf8)
            except Exception as error:
                print("[ERROR TTS] {}".format(error))
        else:
            print("  -> [MOCK TTS SAY] \"{}\"".format(texto_oracion))

        # 3. Gesto cinestetico complementario seguro
        self.ejecutar_animacion_asentimiento()

        # 4. Retorno a estado base de reposo
        self.establecer_leds_base()
        print("[ESTADO] Robot reestablecido en postura base de atencion.\n")

    def apagar_seguro(self):
        """Reestablece el robot y desuscribe la camara antes de finalizar el servidor."""
        print("\n[DESCONEXION] Finalizando sesion del controlador...")
        if self.cam_proxy is not None and self.id_suscriptor_camara is not None:
            try:
                self.cam_proxy.unsubscribe(self.id_suscriptor_camara)
                print("[CAMARA DESCONECTADA] Desuscripcion de ALVideoDevice completada.")
            except Exception as e_cam:
                print("[ERROR DESUSCRIPCION CAMARA] {}".format(e_cam))

        if not self.usar_mock and self.postura is not None:
            try:
                self.postura.goToPosture(POSTURA_BASE, 0.5)
                self.establecer_leds_base()
            except Exception:
                pass


class ServidorVideoNao(object):
    """Servidor TCP en hilo dedicado para transmitir fotogramas de NAOqi a Python 3."""

    def __init__(self, host, puerto, controlador):
        self.host = host
        self.puerto = puerto
        self.controlador = controlador
        self.activo = False
        self.sock_servidor = None
        self.hilo = None

    def iniciar(self):
        """Lanza el hilo de atencion de video."""
        self.activo = True
        self.hilo = threading.Thread(target=self._bucle_servicio)
        self.hilo.daemon = True
        self.hilo.start()
        print("[STREAMING VIDEO] Servidor de video activo en {}:{}".format(self.host, self.puerto))

    def detener(self):
        """Detiene el bucle y cierra el socket del servidor de video."""
        self.activo = False
        if self.sock_servidor:
            try:
                self.sock_servidor.close()
            except Exception:
                pass

    def _bucle_servicio(self):
        self.sock_servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_servidor.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self.sock_servidor.bind((self.host, self.puerto))
            self.sock_servidor.listen(2)
        except Exception as err:
            print("[ERROR BIND VIDEO] No se pudo enlazar socket de video: {}".format(err))
            return

        while self.activo:
            try:
                self.sock_servidor.settimeout(1.0)
                try:
                    cliente, direccion = self.sock_servidor.accept()
                except socket.timeout:
                    continue

                print("[CLIENTE VIDEO CONECTADO] Streaming establecido con {}".format(direccion))
                cliente.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                cliente.settimeout(3.0)

                while self.activo:
                    # Esperar senal de peticion 'F' (Frame request) para latencia minima
                    try:
                        peticion = cliente.recv(1)
                        if not peticion:
                            break
                    except socket.timeout:
                        continue
                    except socket.error:
                        break

                    # Descartar peticiones pendientes en buffer para evitar desfasaje temporal
                    try:
                        cliente.setblocking(False)
                        while True:
                            descarte = cliente.recv(64)
                            if not descarte:
                                break
                    except socket.error:
                        pass
                    finally:
                        cliente.setblocking(True)
                        cliente.settimeout(3.0)

                    datos_frame = self.controlador.obtener_fotograma_actual()
                    if datos_frame is None:
                        continue

                    ancho, alto, canales, esp_color, bytes_img = datos_frame
                    longitud = len(bytes_img)

                    # Cabecera de 20 bytes: ancho, alto, canales, esp_color, longitud
                    cabecera = struct.pack("!IIIII", ancho, alto, canales, esp_color, longitud)
                    try:
                        cliente.sendall(cabecera)
                        cliente.sendall(bytes_img)
                    except socket.error:
                        break

                cliente.close()
                print("[CLIENTE VIDEO DESCONECTADO] Sesion de streaming cerrada.")
            except Exception as e_hilo:
                if self.activo:
                    print("[AVISO STREAM VIDEO] {}".format(e_hilo))


def iniciar_servidor_socket(host, puerto, controlador, servidor_video=None):
    """Inicia el servidor socket TCP y atiende solicitudes entrantes de Python 3."""
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        servidor.bind((host, puerto))
        servidor.listen(2)
        print("=" * 65)
        print(" SERVIDOR PUENTE NAOQI INICIADO (PYTHON 2.7)")
        print(" Escuchando peticiones en: {}:{}".format(host, puerto))
        if servidor_video is not None:
            print(" Streaming de camara activo en: {}:{}".format(servidor_video.host, servidor_video.puerto))
        print(" Destino NAOqi configurado: {}:{}".format(controlador.ip_nao, controlador.puerto_nao))
        print(" Presione Ctrl+C para detener el servidor.")
        print("=" * 65 + "\n")

        while True:
            print("[ESPERANDO] Esperando conexion de control desde Python 3...")
            cliente, direccion = servidor.accept()
            print("[CONECTADO] Cliente de control conectado desde {}:{}".format(direccion[0], direccion[1]))

            bufer_datos = ""
            try:
                while True:
                    datos = cliente.recv(TAMANO_BUFFER)
                    if not datos:
                        print("[CLIENTE DESCONECTADO] Se cerro la conexion del cliente.")
                        break

                    bufer_datos += datos
                    while "\n" in bufer_datos:
                        linea, bufer_datos = bufer_datos.split("\n", 1)
                        linea = linea.strip()
                        if not linea:
                            continue

                        try:
                            solicitud = json.loads(linea)
                            tipo_evento = solicitud.get("tipo", "")

                            if tipo_evento == "TOKEN":
                                token_val = solicitud.get("token", "")
                                controlador.notificar_token_aceptado(token_val)
                                respuesta = json.dumps({"estado": "OK", "evento": "TOKEN_RECIBIDO"}) + "\n"
                                cliente.sendall(respuesta)

                            elif tipo_evento == "ORACION":
                                frase_val = solicitud.get("frase", "")
                                tokens_val = solicitud.get("tokens", [])
                                controlador.verbalizar_oracion_completa(frase_val, tokens_val)
                                respuesta = json.dumps({"estado": "OK", "evento": "ORACION_EJECUTADA"}) + "\n"
                                cliente.sendall(respuesta)

                            elif tipo_evento == "RESET":
                                print("[RESET] Bufer de tokens reiniciado por el usuario.")
                                controlador.establecer_leds_base()
                                respuesta = json.dumps({"estado": "OK", "evento": "RESET_CONFIRMADO"}) + "\n"
                                cliente.sendall(respuesta)

                            elif tipo_evento == "PING":
                                respuesta = json.dumps({"estado": "PONG"}) + "\n"
                                cliente.sendall(respuesta)

                            else:
                                print("[AVISO] Tipo de evento desconocido: {}".format(tipo_evento))

                        except ValueError as err_json:
                            print("[ERROR JSON] No se pudo decodificar el mensaje: {}".format(err_json))

            except socket.error as err_socket:
                print("[AVISO SOCKET] Error en transmision con el cliente: {}".format(err_socket))
            finally:
                cliente.close()

    except KeyboardInterrupt:
        print("\n[INTERRUPCION] Deteniendo el servidor por senal de teclado...")
    finally:
        servidor.close()
        if servidor_video is not None:
            servidor_video.detener()
        controlador.apagar_seguro()
        print("[SERVIDOR DETENIDO] Servidor socket cerrado correctamente.")


def main():
    analizador = argparse.ArgumentParser(
        description="Servidor puente socket para coordinacion del Robot NAO con inferencia LSP."
    )
    analizador.add_argument("--host", default=HOST_SOCKET_DEFECTO, help="Host o IP del socket servidor.")
    analizador.add_argument("--puerto", type=int, default=PUERTO_SOCKET_DEFECTO, help="Puerto TCP para comandos.")
    analizador.add_argument("--puerto-video", type=int, default=PUERTO_VIDEO_DEFECTO, help="Puerto TCP para video.")
    analizador.add_argument("--sin-video", action="store_true", help="Desactivar streaming de video de la camara.")
    analizador.add_argument("--cam-id", type=int, default=CAMARA_SUPERIOR_ID, help="Indice de camara (0: superior, 1: inferior).")
    analizador.add_argument("--cam-res", type=int, default=CAMARA_RESOLUCION_QVGA, help="Resolucion (1: QVGA 320x240, 2: VGA 640x480).")
    analizador.add_argument("--cam-fps", type=int, default=FPS_CAMARA_DEFECTO, help="FPS de captura de la camara (recomendado 15).")
    analizador.add_argument("--nao-ip", default=NAO_IP_DEFECTO, help="IP de NAOqi (Robot fisico o simulador).")
    analizador.add_argument("--nao-puerto", type=int, default=NAO_PUERTO_DEFECTO, help="Puerto de NAOqi.")
    analizador.add_argument("--mock", action="store_true", help="Forzar ejecucion en modo simulacion.")

    argumentos = analizador.parse_args()

    modo_mock = SIMULACION_MOCK or argumentos.mock
    controlador = ControladorRobotNao(
        argumentos.nao_ip,
        argumentos.nao_puerto,
        usar_mock=modo_mock,
        habilitar_camara=not argumentos.sin_video,
        cam_id=argumentos.cam_id,
        cam_res=argumentos.cam_res,
        cam_fps=argumentos.cam_fps
    )

    servidor_video = None
    if not argumentos.sin_video:
        servidor_video = ServidorVideoNao(argumentos.host, argumentos.puerto_video, controlador)
        servidor_video.iniciar()

    iniciar_servidor_socket(argumentos.host, argumentos.puerto, controlador, servidor_video=servidor_video)


if __name__ == "__main__":
    main()
