# ADR 0003: Arquitectura de Comunicación Inter-Proceso (Python 3.12 vs Python 2.7)

## Contexto
El SDK de NAOqi (`pynaoqi`) únicamente ofrece soporte para Python 2.7 (32-bit), impidiendo la importación directa de librerías modernas de Machine Learning como MediaPipe 0.10 y Scikit-Learn 1.9, las cuales requieren Python 3.9+.

## Decisión
Desacoplar la solución en dos entornos aislados comunicados mediante subprocesos o sockets TCP:
* **Módulo de Percepción e Inteligencia (Python 3.12):** Captura video, extrae landmarks, ejecuta inferencia con SVM y decide la acción.
* **Módulo de Actuación del Robot (Python 2.7):** Recibe la orden y comanda los actuadores del robot (`ALMotion`, `ALTextToSpeech`) a través de `pynaoqi`.

## Consecuencias
* **Positivas:** Permite usar las librerías de visión y aprendizaje más avanzadas sin quedar limitados por el entorno heredado de NAOqi.
* **Negativas:** Introduce una leve sobrecarga de comunicación inter-proceso que debe mantenerse por debajo de los 50 ms.
