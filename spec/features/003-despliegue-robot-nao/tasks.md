# Tareas de Implementacion: 003 Despliegue con Robot NAO

- [x] Crear servidor socket `src/robot/bridge_nao_server.py` en Python 2.7 con soporte para proxies de NAOqi.
- [x] Implementar retroalimentacion visual verde de 0.3s por token aceptado (`ALLeds`).
- [x] Implementar accion final ante 'ENVIAR': LEDs azul cian, habla en espanol (`ALTextToSpeech`) y gesto suave (`ALMotion`).
- [x] Integrar controlador Mock simulado de respaldo en consola para pruebas sin robot fisico.
- [x] Configurar puerto por defecto de Choregraphe en 54180.
- [x] Desarrollar cliente e inferencia con maquina de estados `src/robot/app_inferencia_nao.py` en Python 3.12.
- [x] Programar filtro de consistencia temporal (14 frames >= 84%) y enfriamiento anti-duplicados.
- [x] Implementar compositor gramatical fluido en espanol para las 8 clases de LSP.
- [x] Crear interfaz HUD semi-transparente con telemetria en vivo sobre OpenCV.
- [x] Crear atajos de ejecucion en raiz (`bridge_nao_server.py`, `app_inferencia_nao.py`) y en `main.py --nao`.
- [x] Redactar pruebas unitarias en `tests/test_nao_integration.py` con 100% de exito en pytest.
- [x] Auditar ausencia total de la letra enie en los archivos de codigo.
