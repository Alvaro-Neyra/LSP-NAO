# Plan Técnico: 003 - Integración y Despliegue en el Robot NAO

## CÓMO: Arquitectura de Comunicación Inter-Proceso (IPC)
* **Problema:** MediaPipe y Scikit-Learn se ejecutan en Python 3.12 (64-bit), mientras que NAOqi exige Python 2.7 (32-bit).
* **Solución Arquitectónica:**
  * **Opción A (Puente Subproceso):** El script maestro en Python 3 detecta la seña y ejecuta un subproceso ligero que invoca `C:\Python27\python.exe robot_accion.py --seña AYUDA`.
  * **Opción B (Socket TCP Local):** Un servicio servidor persistente en Python 2.7 escucha peticiones en `localhost:9559` y ejecuta acciones en NAOqi sin costo de arranque de proceso.
