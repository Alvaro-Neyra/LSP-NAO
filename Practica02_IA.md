PRÁCTICA N.° 2
Implementación de un Modelo de Machine Learning en el Robot NAO

Curso: Inteligencia artificial  |  Ciclo: 2026-2

Curso: Inteligencia Artificial

1. Descripción

En esta práctica los estudiantes impleme    ntarán un movimiento o modelo dentro del
entorno del robot NAO. El objetivo es integrar los conceptos teóricos del curso con una
aplicación real en robótica, demostrando que el modelo entrenado puede ejecutarse o
interactuar con el robot de forma significativa.

2. Alcance

Cada grupo seleccionará libremente el tipo de movimiento, modelo o acción a
implementar. Lo importante no es la complejidad del modelo, sino la calidad de la
integración con el robot y la solidez del proceso: preprocesamiento de datos,
entrenamiento, evaluación y despliegue.

Algunos ejemplos orientativos:

•  Reconocimiento de gestos o posturas usando los sensores del NAO.
•  Clasificación de comandos de voz.
•  Detección de objetos por color con la cámara del robot.
•  Predicción de una acción motora en función de variables del entorno.

3. Entregables

Los grupos deberán presentar dos entregables:

Entregable 1 — Informe técnico en formato PDF que incluya:

Introducción y motivación.

•
•  Descripción del problema y los datos utilizados.
•  Metodología y modelo elegido con justificación.
•  Resultados y métricas de evaluación.
•  Conclusiones y dificultades encontradas.

Entregable 2 — Video de entre tres y cinco minutos que muestre el modelo
funcionando en el robot NAO, con una breve explicación verbal del equipo.

La fecha máxima de entrega es el domingo 08/11 a las 23.59 hrs.

4. Estructura Detallada del Informe Técnico

El informe debe estar redactado en español, con un mínimo de 8 páginas y máximo de 15,
letra Times New Roman 12, interlineado 1.5, márgenes de 2.5 cm. Se entrega en PDF.

4.1. Portada

Nombre del curso, nombre de la práctica, integrantes del grupo, docente, universidad y
fecha.

4.2. Introducción

•  Contexto general sobre la IA aplicado a robótica.
•  Motivación del problema elegido por el grupo.
•  Objetivos de la práctica: uno general y dos o tres específicos.
•  Alcance y limitaciones del trabajo.

4.3. Marco Teórico

Descripción del algoritmo de IA elegido: fundamento matemático básico,
hiperparámetros principales y criterio de selección frente a otras alternativas. Breve
descripción del robot NAO: arquitectura de sensores, sistema operativo NAOqi y
lenguajes compatibles. No debe ser una copia de Wikipedia; debe evidenciar comprensión
propia.

4.4. Metodología

•  Descripción del problema a resolver.
•  Origen y características del dataset: si es propio, cómo fue recolectado; si es

público, citar la fuente.

•  Preprocesamiento: limpieza, normalización, codificación de variables, manejo de

valores faltantes.

•  División de datos en entrenamiento y prueba.
•  Entrenamiento del modelo y ajuste de hiperparámetros.
•  Estrategia de integración con el NAO: cómo se conecta el modelo con el robot,

qué acción ejecuta el robot según la salida del modelo.

4.5. Resultados

•  Métricas de evaluación: accuracy, F1-score, matriz de confusión o RMSE según

corresponda (si aplica).

•  Al menos una gráfica relevante: curva de aprendizaje, matriz de confusión,

distribución de predicciones, etc (si aplica).

•  Descripción del comportamiento observado del robot durante las pruebas.
•  Comparación entre al menos dos configuraciones del modelo o dos algoritmos (si

aplica).

4.6. Discusión

Interpretación de los resultados obtenidos.

•
•  Limitaciones encontradas durante la implementación.
•  Qué mejoraría el grupo con más tiempo o recursos.

4.7. Conclusiones

Mínimo tres conclusiones concretas, alineadas con los objetivos planteados. No deben ser
resúmenes del trabajo, sino juicios derivados de los resultados.

4.8. Referencias

Formato APA 7.ª edición. Mínimo cinco referencias académicas o técnicas. No se aceptan
páginas de Wikipedia ni blogs sin autoría.

5. Rúbrica de Evaluación

Criterio

Excelente (100%)

Bueno (75%)

Regular (50%)

Deficiente (25%)

Funcionamiento del
modelo en el NAO
Peso: 35%

Calidad técnica del
informe
Peso: 30%

Video de
demostración
Peso: 20%

Creatividad y
dificultad del
problema
Peso: 15%

El modelo corre
correctamente en
el robot, la salida
genera una acción
clara y
reproducible.

Metodología
completa,
resultados con
métricas y
gráficas, redacción
fluida y referencias
correctas.

Video claro,
muestra el modelo
en acción,
explicación oral
coherente y bien
estructurada.

Problema original,
dataset propio o
poco
convencional,
solución no trivial.

Funciona con
limitaciones
menores, la acción
del robot es
perceptible.

Funciona
parcialmente o
solo en
simulación.

No se logró
integración con el
robot.

Metodología clara
pero con alguna
sección
incompleta o
gráficas
insuficientes.

Secciones
importantes
superficiales, sin
métricas
comparativas.

Informe
incompleto o sin
rigor técnico.

Video
comprensible pero
con audio
deficiente o
explicación
parcial.

Problema
interesante con
dataset público
bien justificado.

Se ve el robot pero
no queda claro qué
hace el modelo.

Video ausente,
muy corto o sin
explicación.

Problema básico
con dataset
genérico.

Problema trivial
sin justificación
de elección.

6. Penalizaciones

•  Se descuentan 2 puntos por cada día de retraso en la entrega.
•  Se anula la práctica si se detecta copia entre grupos o uso de código ajeno sin

atribución.

