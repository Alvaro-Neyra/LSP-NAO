# -*- coding: utf-8 -*-
"""Servidor Puente Socket TCP para el Robot Humanoide NAO (Python 2.7).

Recibe comandos de control, tokens aceptados y oraciones armadas desde el
modulo de inferencia en Python 3, coordinando la sintesis de voz (ALTextToSpeech),
gestos cinesteticos (ALMotion, ALRobotPosture) y senalizacion visual (ALLeds).
"""

import argparse
import json
import socket
import sys
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
TAMANO_BUFFER = 4096

NAO_IP_DEFECTO = "127.0.0.1"
NAO_PUERTO_DEFECTO = 56432
POSTURA_BASE = "StandInit"
VELOCIDAD_POSTURA = 0.6


class ControladorRobotNao(object):
    """Encapsula los proxies de NAOqi para coordinar voz, postura, movimiento y luces."""

    def __init__(self, ip_nao, puerto_nao, usar_mock=False):
        self.ip_nao = ip_nao
        self.puerto_nao = int(puerto_nao)
        self.usar_mock = usar_mock
        self.tts = None
        self.postura = None
        self.motion = None
        self.leds = None
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

        except Exception as error:
            print("[ERROR] Falla al conectar con los proxies de NAOqi: {}".format(error))
            print("[FALLBACK] Conmutando a modo simulado Mock para mantener el servidor activo.")
            self.usar_mock = True

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
        """Reestablece el robot antes de finalizar el servidor."""
        print("\n[DESCONEXION] Finalizando sesion del controlador...")
        if not self.usar_mock and self.postura is not None:
            try:
                self.postura.goToPosture(POSTURA_BASE, 0.5)
                self.establecer_leds_base()
            except Exception:
                pass


def iniciar_servidor_socket(host, puerto, controlador):
    """Inicia el servidor socket TCP y atiende solicitudes entrantes de Python 3."""
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        servidor.bind((host, puerto))
        servidor.listen(2)
        print("=" * 65)
        print(" SERVIDOR PUENTE NAOQI INICIADO (PYTHON 2.7)")
        print(" Escuchando peticiones en: {}:{}".format(host, puerto))
        print(" Destino NAOqi configurado: {}:{}".format(controlador.ip_nao, controlador.puerto_nao))
        print(" Presione Ctrl+C para detener el servidor.")
        print("=" * 65 + "\n")

        while True:
            print("[ESPERANDO] Esperando conexion desde el cliente Python 3...")
            cliente, direccion = servidor.accept()
            print("[CONECTADO] Cliente conectado desde {}:{}".format(direccion[0], direccion[1]))

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
        controlador.apagar_seguro()
        print("[SERVIDOR DETENIDO] Servidor socket cerrado correctamente.")


def main():
    analizador = argparse.ArgumentParser(
        description="Servidor puente socket para coordinacion del Robot NAO con inferencia LSP."
    )
    analizador.add_argument("--host", default=HOST_SOCKET_DEFECTO, help="Host o IP del socket servidor.")
    analizador.add_argument("--puerto", type=int, default=PUERTO_SOCKET_DEFECTO, help="Puerto TCP del socket.")
    analizador.add_argument("--nao-ip", default=NAO_IP_DEFECTO, help="IP de NAOqi (Robot fisico o simulador).")
    analizador.add_argument("--nao-puerto", type=int, default=NAO_PUERTO_DEFECTO, help="Puerto de NAOqi.")
    analizador.add_argument("--mock", action="store_true", help="Forzar ejecucion en modo simulacion.")

    argumentos = analizador.parse_args()

    modo_mock = SIMULACION_MOCK or argumentos.mock
    controlador = ControladorRobotNao(argumentos.nao_ip, argumentos.nao_puerto, usar_mock=modo_mock)
    iniciar_servidor_socket(argumentos.host, argumentos.puerto, controlador)


if __name__ == "__main__":
    main()
