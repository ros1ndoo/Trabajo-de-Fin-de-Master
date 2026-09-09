17-06-2026

# Predicción de Fiabilidad y Valor en Lanzamientos Automovilísticos

Este repositorio contiene el desarrollo de un proyecto de Data Science enfocado en resolver la asimetría de información en el mercado automotriz. A través de la extracción, ingeniería y modelado de datos, el proyecto culmina en un dashboard interactivo que estima la **propensión a recalls de seguridad** de nuevos vehículos basándose en su historial técnico y en el comportamiento previo del fabricante.

> ⚠️ **Nota importante sobre la variable objetivo:** El índice producido por este proyecto es un **proxy construido a partir de registros oficiales de *recalls* de la NHTSA**, no una medida directa de fiabilidad mecánica general del día a día. Un *recall* de seguridad no equivale a una avería mecánica cotidiana. Esta distinción se mantiene explícita en todas las capas del proyecto.

---

## 1. El Problema y la Solución

**El Problema:** Al adquirir un vehículo, los consumidores e inversores de flotas toman decisiones guiados principalmente por campañas de marketing. Existe una profunda asimetría de información respecto a la propensión a fallos de seguridad que un modelo puede manifestar, información que solo se conoce empíricamente tras años de rodaje en el mercado.

**La Solución:** Un modelo predictivo que anticipa el índice de propensión a *recalls* de un vehículo antes de que acumule años en el mercado. El modelo se apoya en:
- Especificaciones técnicas del vehículo (cilindrada, potencia, categoría).
- Historial de *recalls* del fabricante en los **3 años previos al lanzamiento** que se predice.

> **Alcance del Proyecto:** Para garantizar la consistencia analítica y la precisión de los datos oficiales, el alcance está acotado estrictamente al **mercado estadounidense** y a vehículos fabricados entre **1995 y la actualidad**.

---

## 2. Arquitectura y Fuentes de Datos

El pipeline de datos sigue una arquitectura de medallón (**Raw → Processed → Gold**) integrando dos fuentes concretas y cerradas:

1. **NHTSA API (National Highway Traffic Safety Administration):**
   - **Tipo:** API REST (JSON).
   - **Aporte:** Registros oficiales gubernamentales de *recalls* por marca, modelo y año. Solo se utiliza la **ventana de los primeros 3 años** desde el lanzamiento de cada modelo para asegurar la comparabilidad entre cohortes de distintos años.

2. **Kaggle — "Car Features and MSRP" (CooperUnion):**
   - **Tipo:** Fichero estático (CSV).
   - **Aporte:** Especificaciones técnicas detalladas (cilindrada, potencia, categoría, precio de lanzamiento) filtradas para el mercado norteamericano. Fuente identificada y estable.

### Estructura del repositorio

```
├── data/
│   ├── raw/          # Datos originales inmutables (JSON de NHTSA + CSV de Kaggle)
│   ├── processed/    # SQLite transaccional para limpieza y cruce de fuentes
│   └── gold/         # gold_us_car_reliability.parquet — dataset final listo para ML
├── docs/
│   ├── entregas/     # Documentación de diseño y decisiones de arquitectura
│   └── assets/       # Mockup del frontal 
├── notebooks/        # EDA y experimentación
└── src/              # Scripts Python de producción (ingesta, limpieza, modelo)
```

---

## 3. Variable Objetivo: `indice_fiabilidad_100`

La variable objetivo se construye en tres fases para garantizar su comparabilidad y defendibilidad:

1. **Ponderación por gravedad de recall** usando palabras clave en la descripción NHTSA (Crítico ×3.0 / Moderado ×1.5 / Leve ×1.0).
2. **Ventana de observación fija de 3 años:** Solo se contabilizan los *recalls* emitidos durante los primeros 3 años desde el lanzamiento. Solo participan en el entrenamiento los modelos que hayan completado dicha ventana. Esto hace comparable un coche de 2005 con uno de 2022.
3. **Normalización Z-score intra-segmento:** Calculada **exclusivamente sobre el conjunto de entrenamiento** y aplicada después al resto de conjuntos para evitar data leakage. El resultado se escala a un rango de 0 a 100.

### Anti-leakage garantizado

- Los *recalls* del propio vehículo consultado **no entran como features** de predicción.
- La variable `hist_fiabilidad_marca` se construye únicamente con datos conocidos **antes** del lanzamiento del modelo que se está prediciendo.
- El Z-score del target no se calcula con todo el dataset antes de separar train y test.

---

## 4. Stack Tecnológico

### Ingesta y Procesamiento (Data Engineering)
- **Requests & JSON:** Extracción por lotes (*batch*) a la API de la NHTSA gestionando *rate limiting*.
- **Pandas & NumPy:** Manipulación, limpieza y Feature Engineering.
- **TheFuzz:** *Fuzzy Matching* (`token_set_ratio`) para resolver inconsistencias semánticas entre nomenclaturas de Kaggle y la NHTSA. Los cruces quedan auditados en `audit_fuzzy_matches.csv` con umbrales: Auto ≥90 / Revisión manual 75–89 / Rechazo <75.
- **SQLite3:** Motor relacional intermedio para transformaciones SQL sobre el dataset unificado.

### Modelado Predictivo (Machine Learning)
- **Scikit-Learn:** Pipelines de preprocesamiento (escalado, One-Hot / Target Encoding) y validación temporal estricta (Train 1995–2018 / Valid 2019–2021 / Test 2022+).
- **Ridge Regression:** Modelo principal por su interpretabilidad mediante coeficientes.
- **XGBoost / Random Forest:** Alternativa avanzada; seleccionada solo si supera a Ridge en >10% de reducción de MAE.
- **SHAP:** Explicabilidad del modelo avanzado si se selecciona.

### Visualización y Producto Final (MVP)
- **Matplotlib & Seaborn:** EDA.
- **Streamlit + Plotly:** Framework de despliegue del dashboard interactivo MVP.
- **Bootstrap 5 + Chart.js:** Mockup de alta fidelidad del frontal (modo oscuro). Ver [`docs/assets/mockup.html`](docs/assets/mockup.html).

---

## 5. Plan de Contingencia

Si la tasa de pérdida de modelos tras el *Fuzzy Matching* supera el **40%** (registros huérfanos), el proyecto prescindirá de la variable de *recalls* de la API. La alternativa directa será predecir la **depreciación económica del vehículo**, cruzando el MSRP de Kaggle con precios de segunda mano para calcular la pérdida de valor según especificaciones técnicas.

---

## 6. Roadmap

1.  Definición del problema, fuentes y viabilidad *(Entrega 02)*
2.  Diseño del modelo de datos y capa gold *(Entrega 03)*
3.  Diseño del análisis y estrategia de modelado *(Entrega 04)*
4.  Mockup del frontal Bootstrap 5 modo oscuro *(Entrega 05)*
5.  Ingesta masiva y script de extracción automatizada de la API NHTSA
6.  Pipeline de limpieza y *Fuzzy Matching* auditado
7.  Consolidación de la capa Gold en Parquet
8.  Análisis Exploratorio de Datos (EDA)
9.  Entrenamiento, validación temporal y ajuste del modelo
10.  Despliegue del dashboard en Streamlit
