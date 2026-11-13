"""Script de Verificacion y Diagnostico de la Camara del Robot NAO.

Permite comprobar de forma rapida y aislada que el streaming de video desde
el servidor puente en Python 2.7 (ALVideoDevice) este funcionando con fluidez
y sin retardos en la laptop.
"""

import argparse
import socket
import struct
import sys
import time
import cv2
import numpy as np


class ClienteVisorCamaraNao:
    """Receptor TCP para diagnostico del flujo de video de NAO."""

    def __init__(self, host: str = "127.0.0.1", puerto: int = 50006) -> None:
        self.host = host
        self.puerto = puerto
        self.sock: socket.socket | None = None
        self.conectado = False
        self.conectar()

    def conectar(self) -> bool:
        """Establece conexion con el puerto de streaming."""
        try:
            if self.sock:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(3.0)
            self.sock.connect((self.host, self.puerto))
            self.conectado = True
            print(f"[CONECTADO] Conectado al stream de video de NAO en {self.host}:{self.puerto}")
            return True
        except (socket.error, socket.timeout) as error:
            print(f"[ERROR] No se pudo conectar a {self.host}:{self.puerto} -> {error}")
            self.conectado = False
            self.sock = None
            return False

    def obtener_fotograma(self) -> tuple[bool, np.ndarray | None]:
        """Solicita y recibe un fotograma reciente."""
        if not self.conectado or self.sock is None:
            if not self.conectar():
                return False, None

        try:
            self.sock.sendall(b"F")

            cabecera = self._recibir_exacto(20)
            if not cabecera:
                self.conectado = False
                return False, None

            ancho, alto, canales, esp_color, longitud = struct.unpack("!IIIII", cabecera)
            datos_raw = self._recibir_exacto(longitud)
            if not datos_raw:
                self.conectado = False
                return False, None

            cuadro = np.frombuffer(datos_raw, dtype=np.uint8).reshape((alto, ancho, canales)).copy()
            if esp_color == 11:
                cuadro = cv2.cvtColor(cuadro, cv2.COLOR_RGB2BGR)

            if ancho < 640:
                cuadro = cv2.resize(cuadro, (640, 480), interpolation=cv2.INTER_LINEAR)

            return True, cuadro
        except (socket.error, socket.timeout):
            self.conectado = False
            return False, None

    def _recibir_exacto(self, n_bytes: int) -> bytes | None:
        bufer = bytearray()
        while len(bufer) < n_bytes:
            fragmento = self.sock.recv(min(n_bytes - len(bufer), 65536)) if self.sock else b""
            if not fragmento:
                return None
            bufer.extend(fragmento)
        return bytes(bufer)

    def cerrar(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.conectado = False


def main() -> None:
    analizador = argparse.ArgumentParser(description="Visor de Diagnostico de Camara NAO")
    analizador.add_argument("--host", default="127.0.0.1", help="Host del stream de video.")
    analizador.add_argument("--puerto", type=int, default=50006, help="Puerto del stream de video.")
    argumentos = analizador.parse_args()

    print("=" * 65)
    print(" VISOR DE DIAGNOSTICO - CAMARA DEL ROBOT NAO")
    print(f" Conectando a {argumentos.host}:{argumentos.puerto}...")
    print(" Presione [Q] en la ventana de video para salir.")
    print("=" * 65)

    visor = ClienteVisorCamaraNao(argumentos.host, argumentos.puerto)
    if not visor.conectado:
        print("\n[ERROR] Verifique que 'bridge_nao_server.py' este corriendo en Python 2.7.")
        return

    tiempo_anterior = time.time()
    try:
        while True:
            exito, fotograma = visor.obtener_fotograma()
            if not exito or fotograma is None:
                print("[REINTENTO] Esperando datos de video de NAO...")
                time.sleep(0.1)
                continue

            ahora = time.time()
            fps = 1.0 / max(ahora - tiempo_anterior, 1e-4)
            tiempo_anterior = ahora

            alto, ancho = fotograma.shape[:2]
            cv2.putText(
                fotograma, f"NAO POV | {ancho}x{alto} | FPS: {fps:.1f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                fotograma, "Presione [Q] para salir",
                (20, alto - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )

            cv2.imshow("Visor de Camara NAO - Diagnostico", fotograma)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        visor.cerrar()
        cv2.destroyAllWindows()
        print("\nVisor de diagnostico cerrado.")


if __name__ == "__main__":
    main()
