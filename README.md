17-06-2026

# Predicción de Fiabilidad y Valor en Lanzamientos Automovilísticos

Este repositorio contiene el desarrollo de un proyecto de Data Science enfocado en resolver la asimetría de información en el mercado automotriz. A través de la extracción, ingeniería y modelado de datos, el proyecto culmina en un dashboard interactivo que estima la fiabilidad mecánica de nuevos vehículos basándose en su herencia técnica.

---

## 1. El Problema y la Solución
**El Problema:** Al adquirir un vehículo, los consumidores e inversores de flotas toman decisiones guiados principalmente por campañas de marketing y la estética del producto. Existe una profunda asimetría de información respecto a los fallos mecánicos reales que un modelo o familia de motores puede desarrollar con el tiempo.

**La Solución:** Un modelo predictivo que anticipa la fiabilidad de un vehículo y/o su depreciación de mercado antes de que acumule años de rodaje. El modelo se apoya en datos históricos de especificaciones técnicas y registros oficiales de llamadas a revisión (*recalls*). 

> **Alcance del Proyecto:** Para garantizar la consistencia analítica y la precisión de los datos oficiales, el alcance de este proyecto está acotado estrictamente al mercado estadounidense y a vehículos fabricados entre 1995 y la actualidad.

---

## 2. Arquitectura y Fuentes de Datos

El pipeline de datos sigue una arquitectura de medallón (Raw -> Processed -> Gold) integrando dos fuentes principales:

1. **NHTSA API (National Highway Traffic Safety Administration):** * **Tipo:** API REST (JSON).
   * **Aporte:** Registros oficiales gubernamentales de quejas y llamadas a revisión (Recalls) por marca, modelo y año.
2. **Kaggle Datasets (Car Features & MSRP):**
   * **Tipo:** Ficheros estáticos (CSV).
   * **Aporte:** Especificaciones técnicas detalladas (cilindrada, potencia, tracción, tipo de combustible, precio de lanzamiento).

### Estructura del repositorio:
* `/data/raw/`: Datos originales inmutables (JSON y CSV).
* `/data/processed/`: Base de datos SQLite transaccional para limpieza y cruce de datos.
* `/data/gold/`: Archivos `.parquet` consolidados para el entrenamiento del modelo.
* `/docs/entregas/`: Documentación de diseño y arquitectura del proyecto.
* `/notebooks/`: Entornos de experimentación y Análisis Exploratorio de Datos (EDA).
* `/src/`: Scripts de Python de producción para la ingesta, limpieza y entrenamiento.

---

## 3. Stack Tecnológico

El proyecto está desarrollado íntegramente en Python 3.x, empleando el siguiente ecosistema de librerías y tecnologías:

### Ingesta y Procesamiento de Datos (Data Engineering)
* **Requests & JSON:** Para la extracción de datos mediante peticiones por lotes (batch) a la API de la NHTSA, gestionando políticas de *rate limiting*.
* **Pandas & NumPy:** Para la manipulación vectorial, limpieza de datos y Feature Engineering.
* **TheFuzz (FuzzyWuzzy):** Crítico para la integración de fuentes. Se emplea *Fuzzy Matching* basado en la distancia de Levenshtein para resolver las inconsistencias semánticas en la nomenclatura de modelos entre la base de datos estática y la gubernamental.
* **SQLite3:** Motor relacional intermedio para ejecutar transformaciones SQL complejas sobre el *dataframe* unificado.

### Modelado Predictivo (Machine Learning)
* **Scikit-Learn:** Para la construcción de pipelines de preprocesamiento (escalado, codificación de variables categóricas) y validación cruzada.
* **XGBoost / RandomForest:** Algoritmos de ensamblaje (basados en árboles) seleccionados por su robustez para modelar relaciones no lineales entre especificaciones mecánicas y el índice de averías.

### Visualización y Producto Final
* **Matplotlib & Seaborn:** Para el Análisis Exploratorio de Datos (EDA).
* **Streamlit:** Framework de despliegue para materializar el Producto Mínimo Viable (MVP). Construirá la interfaz web analítica para la consulta de predicciones.

---

## 4. Próximos Pasos (Roadmap)
1. Ingesta masiva de datos estáticos y diseño del script de extracción automatizada para la API.
2. Ejecución del pipeline de limpieza y algoritmos de Fuzzy Matching.
3. Consolidación de la capa Gold en formato Parquet.
4. Análisis Exploratorio de Datos (EDA).
5. Entrenamiento, validación y ajuste de hiperparámetros del modelo predictivo.
6. Despliegue en producción del dashboard interactivo.
