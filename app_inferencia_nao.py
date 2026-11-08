"""Lanzador raiz para la aplicacion de inferencia y control de NAO (Python 3)."""
import os
import subprocess
import sys

ruta_destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "robot", "app_inferencia_nao.py")

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, ruta_destino] + sys.argv[1:]))

