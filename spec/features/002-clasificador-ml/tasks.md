# Tareas: 002 - Entrenamiento y Comparativa de Modelos ML

- [x] Implementar carga y verificación de dimensiones del dataset consolidado ($12,046$ filas, $126$ características).
- [x] Configurar división estratificada `train_test_split(test_size=0.20, random_state=42)`.
- [x] Configurar Pipeline SVM (`StandardScaler` + `SVC(C=10.0, rbf, probability=True)`).
- [x] Configurar `RandomForestClassifier(n_estimators=100, n_jobs=-1)`.
- [x] Medir tiempos de entrenamiento y latencia de inferencia por muestra.
- [x] Generar reporte tabular `reports/metricas_comparativas.txt`.
- [x] Exportar figuras editoriales a 300 DPI (`reports/figures/`).
- [x] Serializar el modelo ganador (`models/modelo_lsp_ganador.joblib`).
