# Contrato de Datos (Data Contract) - LSP Bimanual

## 1. Esquema del Vector de Características (X)
Cada muestra capturada debe contener exactamente **126 características numéricas de tipo flotante**, distribuidas en dos ranuras de lateralidad fijas:

| Bloque | Columnas | Descripción |
| :--- | :--- | :--- |
| **Mano Izquierda (`lh_`)** | `lh_lm_0_x` a `lh_lm_20_z` (63 valores) | Coordenadas normalizadas de los 21 puntos clave de la mano izquierda. |
| **Mano Derecha (`rh_`)** | `rh_lm_0_x` a `rh_lm_20_z` (63 valores) | Coordenadas normalizadas de los 21 puntos clave de la mano derecha. |

### Reglas de Relleno (Zero-Padding) y Normalización:
* **Invarianza por traslación:** Cada mano se centra restando las coordenadas de su muñeca:
  $$\vec{p}_{i}^{rel} = \vec{p}_{i} - \vec{p}_{\text{muñeca}}$$
* **Invarianza por escala:** Las coordenadas relativas se dividen entre la distancia euclidiana máxima o el valor absoluto máximo:
  $$\vec{p}_{i}^{norm} = \frac{\vec{p}_{i}^{rel}}{\max(|\vec{p}^{rel}|, 10^{-6})}$$
* **Mano ausente:** Si un gesto es unimanual o una mano no es detectada, las 63 posiciones de dicha mano se rellenan obligatoriamente con ceros (`0.0`).

---

## 2. Esquema de Salida y Metadatos (y)

| Columna | Tipo | Rango / Valores | Descripción |
| :--- | :--- | :--- | :--- |
| **`label`** | Entero (`int64`) | `0` a `7` | Identificador de la clase asignada. |
| **`class_name`** | Cadena (`str`) | `YO`, `TU`, `NECESITAR`, `QUERER`, `AGUA`, `AYUDA`, `MEDICINA`, `ENVIAR` | Nombre oficial de la seña en LSP. |
| **`integrante`** | Cadena (`str`) | `alvaro`, `arturo`, `eduardo`, `rai`, `victor` | Trazabilidad del sujeto participante. |

---

## 3. Criterios de Aceptación del Dataset
* Cero valores nulos (`NaN` o `None`).
* Exactamente 128 columnas en datasets de entrenamiento y 129 columnas al incluir el metadato `integrante`.
* Desbalance de clases menor al 5% entre cualquier par de señas.
