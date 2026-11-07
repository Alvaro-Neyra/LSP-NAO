# ADR 0002: Elección de Support Vector Machine (RBF) frente a Random Forest

## Contexto
Se evaluaron dos arquitecturas clásicas de Machine Learning sobre el corpus de 12,046 muestras: Support Vector Machine con kernel RBF y Random Forest Classifier con 100 estimadores. Se requería balancear exactitud global, F1-Score, tamaño en memoria y latencia de inferencia para despliegue en el robot NAO.

## Decisión
Seleccionar **Support Vector Machine (RBF)** como el modelo de producción para el robot NAO.

## Justificación Basada en Evidencia Empírica
1. **Desempeño predictivo:** SVM obtuvo una Exactitud de **96.02%** y un F1-Macro de **96.03%**, superando a Random Forest (94.94% y 94.94% respectivamente).
2. **Latencia crítica en robótica:** SVM alcanzó una latencia de **0.408 ms por muestra**, siendo **~37 veces más rápido** que Random Forest (15.248 ms por muestra).
3. **Huella de memoria:** El modelo SVM serializado ocupa **3.07 MB**, casi 6 veces menos que los **17.85 MB** requeridos por los árboles de decisión de Random Forest.

## Consecuencias
* **Positivas:** El robot NAO puede realizar inferencia en tiempo real sin experimentar demoras en la respuesta motriz o auditiva.
* **Negativas:** El tiempo de entrenamiento de SVM (4.42 s) fue ligeramente superior al de Random Forest (0.46 s), lo cual es irrelevante dado que el entrenamiento se ejecuta fuera de línea.
