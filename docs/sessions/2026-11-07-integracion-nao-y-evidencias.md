# Bitacora de Sesion: 2026-11-07

## Resumen de Actividades
1. **Banco de Pruebas en Vivo y Captura de Evidencias (`src/vision/probar_modelo_en_vivo.py`):**
   - Implementacion del disparador automatico por estabilidad temporal (dwell-time de 0.75 s a >= 85% de certeza).
   - Estampado visual de tarjeta diagnostica: clase detectada, confianza %, latencia y recuadro envolvente.
   - Generador automatico de galeria en cuadricula interactiva con Matplotlib al finalizar la sesion (tecla Q).
   - Verificacion y cumplimiento estricto de la regla de caracteres: 0 ocurrencias de la letra enie en el codigo.

2. **Despliegue y Pruebas en Google Colab (`notebooks/banco_pruebas_lsp_colab.ipynb`):**
   - Transmision de camara web en vivo dentro del cuaderno mediante puente JavaScript/WebRTC y evaluacion continua.
   - Generador de mosaico grafico final con todas las capturas recolectadas.

3. **Arquitectura de Integracion Cliente-Servidor con Robot NAO:**
   - **Servidor Puente (`src/robot/bridge_nao_server.py` en Python 2.7 / PyNAOqi):**
     - Proxies `ALTextToSpeech` (idioma Spanish), `ALRobotPosture` (`StandInit`), `ALMotion` y `ALLeds`.
     - Feedback tactico inmediato: ojos verdes (0.3 s) por token individual aceptado.
     - Ejecucion final con seña ENVIAR: ojos azul cian, vocalizacion de la frase armada y gesto cinestetico de asentimiento.
     - Controlador Mock de respaldo para ejecucion y pruebas locales sin robot fisico.
     - Configuracion por defecto del puerto virtual de Choregraphe: 54180.
   - **Inferencia y Maquina de Estados (`src/robot/app_inferencia_nao.py` en Python 3.12):**
     - Filtro temporal de consistencia (14 frames continuos a >= 84%) y enfriamiento anti-duplicados.
     - Compositor linguistico para transformar secuencias de tokens en oraciones gramaticales naturales.
     - Transmision por socket TCP no bloqueante (puerto 50005) hacia el puente en Python 2.7.
   - Atajos lanzadores directos en raiz y nueva opcion en el orquestador `main.py --nao`.

4. **Suite de Pruebas Unitarias (`tests/test_nao_integration.py`):**
   - 9 pruebas automatizadas en total con pytest ejecutandose exitosamente (100% passing).
