# Especificación: 002 - Entrenamiento y Comparativa de Modelos ML

## QUÉ: Requerimientos Funcionales y Criterios Medibles
1. **Modelos a evaluar:** Support Vector Machine con kernel RBF vs Random Forest Classifier.
2. **Partición de datos:** División estratificada 80% entrenamiento y 20% prueba (`random_state=42`).
3. **Métricas objetivo:**
   * Exactitud global (*Accuracy*) $> 95\%$.
   * F1-Score Macro $> 95\%$.
   * Latencia promedio de inferencia por muestra $< 5.0\text{ ms}$.
   * Tamaño del modelo serializado en disco $< 20\text{ MB}$.
4. **Entregables gráficos y tabulares:**
   * Matriz de confusión dual normalizada en porcentaje a 300 DPI.
   * Gráfica de barras comparativa de F1 por clase a 300 DPI.
   * Reporte de texto exportable a la Sección 5 del informe técnico.
