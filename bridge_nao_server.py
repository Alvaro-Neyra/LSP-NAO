# -*- coding: utf-8 -*-
"""Lanzador raiz para el servidor puente de NAOqi (Python 2.7)."""
import os
import subprocess
import sys

ruta_destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "robot", "bridge_nao_server.py")

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, ruta_destino] + sys.argv[1:]))

