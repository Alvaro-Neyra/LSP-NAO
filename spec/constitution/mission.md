# Misión y Objetivos del Proyecto

## Propósito
Desarrollar un sistema de Inteligencia Artificial capaz de reconocer en tiempo real **Lengua de Señas Peruana (LSP)** a partir de la transmisión de video, integrando la inferencia del modelo con el robot humanoide **NAO** para que este actúe como un intérprete o interlocutor robótico inclusivo mediante síntesis de voz y gestos articulares.

---

## Objetivos del Proyecto

### Objetivo General
Implementar y desplegar un modelo de Machine Learning supervisado en el entorno operativo del robot NAO que clasifique con alta exactitud señas de LSP y ejecute respuestas motoras y auditivas coherentes.

### Objetivos Específicos
1. **Recolección Colaborativa:** Construir un dataset bimanual robusto con 8 clases representativas de LSP, balanceado con 300 muestras por seña recolectadas de 5 sujetos distintos ($>12,000$ muestras).
2. **Evaluación Comparativa de Modelos:** Comparar experimentalmente algoritmos de aprendizaje estadístico (Support Vector Machine y Random Forest) priorizando la exactitud global y la latencia sub-milisegundo.
3. **Integración con NAOqi:** Establecer una arquitectura de comunicación desacoplada y eficiente entre el entorno moderno de inferencia (Python 3.12) y el runtime de control del robot (Python 2.7).
4. **Validación Experimental:** Garantizar una exactitud superior al 95% y una latencia de inferencia menor a $5\text{ ms}$, documentando el proceso con rigor científico en el informe técnico y demostración en video.
