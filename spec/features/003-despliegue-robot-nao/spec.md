# 003: Despliegue e Integracion con Robot Humanoide NAO

## 1. Descripcion General
Integrar el clasificador de señas LSP (Python 3.12) con el robot humanoide NAO (Python 2.7 / NAOqi 2.1.4) mediante una arquitectura de comunicacion desacoplada basada en Sockets TCP locales.

## 2. Requerimientos Funcionales
* **RF-001 (Servidor Puente Python 2.7):** Inicializar proxies `ALTextToSpeech` ('Spanish'), `ALRobotPosture` ('StandInit'), `ALMotion` y `ALLeds` en `127.0.0.1:54180` (o IP física).
* **RF-002 (Feedback Táctico Inmediato):** Al recibir un token aceptado por socket, cambiar los LEDs faciales a verde por 0.3 segundos.
* **RF-003 (Ejecución Final de Oración):** Al recibir la seña 'ENVIAR', vocalizar la oración completa armada en español y ejecutar una animación cinestética suave de asentimiento y brazos.
* **RF-004 (Máquina de Estados y Debounce):** Requerir 14 fotogramas consecutivos (>= 84% confianza) para validar un token y aplicar enfriamiento de 2.0 s.
* **RF-005 (Compositor Gramatical):** Transformar secuencias de tokens (ej. `['YO', 'NECESITAR', 'AGUA']`) en oraciones naturales (*"Yo necesito agua."*).
* **RF-006 (HUD Diagnóstico):** Desplegar interfaz en video con seña actual, barra de consistencia, búfer y estado del socket.

## 3. Criterios de Aceptacion
* Comunicación Socket TCP estable en `127.0.0.1:50005` con tolerancia a desconexiones temporales.
* Inclusión de un modo simulado Mock en el servidor si no hay conexión con el robot.
* 0 ocurrencias de la letra enie en el código Python.
