# Entrega 3 - Diseño del modelo de datos y capa gold

## 1. Resumen de la idea y datos del proyecto
El proyecto busca resolver el problema de asimetría de información al que se enfrentan los consumidores al adquirir vehículos, basando las decisiones de compra en datos empíricos de fiabilidad en lugar de marketing.

La solución es construir un modelo predictivo (y un dashboard interactivo) que estime la fiabilidad de lanzamientos recientes analizando el historial mecánico de la marca y las especificaciones técnicas del motor. Para garantizar la viabilidad del cruce de datos, **el proyecto se acotará estrictamente al mercado estadounidense para vehículos fabricados desde 1995 hasta la actualidad**. 

Se utilizarán dos fuentes de datos principales:
* **API de la NHTSA (National Highway Traffic Safety Administration):** Proporcionará los registros oficiales de fallos mecánicos y *recalls* en EE. UU. (A partir de 1995, la NHTSA garantiza un 99% de precisión en la decodificación de datos).
* **Datasets de Kaggle:** Aportarán las bases de datos con especificaciones técnicas (cilindrada, potencia, etc.) filtradas para el mercado norteamericano.

---

## 2. Tecnología o formato de almacenamiento elegido
El ecosistema de datos requerirá un almacenamiento híbrido para procesar diferentes formatos antes de consolidarlos para el Machine Learning:

* **Ficheros JSON y CSV (Capa Raw):** Las peticiones a la API de la NHTSA se descargarán inicialmente en JSON (utilizando peticiones por lotes/batch para evitar bloqueos por *rate limits*), mientras que los datos estáticos de Kaggle se ingerirán en CSV.
* **Base de datos relacional (SQLite):** Se utilizará como motor de procesamiento intermedio. Permitirá realizar cruces complejos (*joins* y agregaciones) mediante SQL sin desbordar la memoria RAM del entorno de desarrollo.
* **Ficheros Parquet (Capa Gold):** Formato final elegido para almacenar el dataset analítico. Se justifica por su compresión eficiente, su estructura columnar que acelera las consultas para dashboards y su capacidad para preservar de forma estricta los tipos de datos en Python/Pandas.

---

## 3. Estructura de capas de datos
El repositorio seguirá una arquitectura de medallón estructurada en los siguientes directorios lógicos:

* `data/raw/`: Datos originales e inmutables. JSONs devueltos por la API de la NHTSA y los CSVs técnicos de Kaggle.
* `data/processed/`: Datos intermedios donde se aplicarán filtros (ej. exclusión de vehículos anteriores a 1995), imputación de nulos y estandarización de textos.
* `data/gold/`: Dataset final (`.parquet`) que contendrá el cruce exitoso de características técnicas e índices de fallos, listo para entrenar el modelo predictivo.

---

## 4. Definición de la capa gold
El entregable final de la preparación de datos será una tabla maestra que actuará como contrato de datos para el análisis y modelado.

* **Nombre del dataset:** `gold_us_car_reliability.parquet`
* **Descripción funcional:** Tabla consolidada que vincula las especificaciones mecánicas de cada vehículo con su índice histórico de *recalls* en EE. UU.
* **Nivel de granularidad:** Una fila por marca, modelo, motorización y año.
* **Número esperado de registros:** Entre 30.000 y 80.000 registros (condicionado por el filtro temporal de 1995-actualidad).
* **Clave primaria:** Clave compuesta (`id_marca_modelo_motor_ano`).
* **Variable objetivo:** `indice_fiabilidad` (variable numérica ponderada basada en la cantidad de llamadas a revisión).
* **Uso posterior:** Análisis Exploratorio de Datos (EDA), entrenamiento del modelo de regresión/clasificación y consumo en el dashboard final.

---

## 5. Relaciones entre datos
El mayor desafío técnico es cruzar la base de datos técnica (Kaggle) con los reportes de incidentes (NHTSA).

* **Claves lógicas:** La relación se establecerá mediante la combinación de `Marca` + `Modelo` + `Año`.
* **Tipo de relación (N:M):** Un mismo motor se usa en diferentes modelos, y un mismo modelo experimenta múltiples *recalls* independientes a lo largo del tiempo.
* **Técnica de cruce:** Debido a la inconsistencia esperada en los nombres comerciales (ej. "F-150" en Kaggle vs "F150" en la NHTSA), no se utilizarán `JOINs` tradicionales estrictos. Se aplicarán técnicas de **Fuzzy Matching** (búsqueda difusa mediante librerías como *FuzzyWuzzy* o *TheFuzz*) para maximizar la tasa de emparejamiento entre ambas fuentes. Además, se agregarán previamente los *recalls* por modelo y año antes de ejecutar el cruce.

---

## 6. Diccionario de datos inicial

| Campo | Descripción | Tipo de dato | Fuente | Obligatorio | Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id_vehiculo` | Identificador único compuesto | string | Derivado | Sí | Formato: MARCA_MODELO_AÑO |
| `marca` | Fabricante del vehículo | string | Kaggle/NHTSA | Sí | Normalizado a minúsculas |
| `modelo` | Nombre comercial en EE. UU. | string | Kaggle/NHTSA | Sí | Procesado con Fuzzy Matching |
| `ano_fabricacion` | Año de lanzamiento | int | Kaggle | Sí | Rango estricto: >= 1995 |
| `cilindrada_cc` | Volumen del motor | int | Kaggle | Sí | |
| `potencia_cv` | Potencia (Horsepower) | float | Kaggle | No | Imputación si es nulo |
| `num_recalls` | Cantidad total de llamadas a revisión | int | NHTSA | No | Si no cruza, se asume 0 o nulo controlable |
| `indice_fiabilidad` | Variable objetivo predictiva | float | Derivado | Sí | Calculada post-agregación |

---

## 7. Problemas de calidad esperados
Al integrar fuentes dispares, se anticipan los siguientes riesgos de calidad que deberán ser mitigados mediante código:

* **Throttling y límites de la API:** Las peticiones continuas a la NHTSA pueden provocar bloqueos de IP (*rate limits*), lo que requerirá implementar pausas automáticas (`time.sleep`) o peticiones por lotes en la extracción.
* **Discrepancia semántica (Nomenclaturas):** Diferencias menores en la forma de escribir los modelos incluso dentro del mismo mercado estadounidense.
* **Evolución del registro de fallos:** Cambios en la taxonomía que usa la NHTSA para definir la "gravedad" de un *recall* entre los años 1995 y 2024.
* **Valores nulos en especificaciones:** Fichas técnicas incompletas en Kaggle para vehículos minoritarios o ediciones limitadas.

---

## 8. Decisiones de limpieza y transformación previstas
Se establecerán las siguientes reglas en el *pipeline* de datos hacia la capa processed:

* **Filtro temporal estricto:** Eliminación inmediata de cualquier registro donde `ano_fabricacion < 1995`.
* **Normalización de *Strings*:** Transformación de texto a minúsculas, eliminación de signos de puntuación y aplicación de *Fuzzy Matching* con un umbral de similitud mínimo del 85% para considerar que dos modelos coinciden.
* **Tratamiento de nulos:** Se imputará la mediana de la marca/categoría para variables técnicas continuas faltantes (ej. potencia). Si un registro carece de año o modelo, será eliminado.
* **Feature Engineering:** Creación de variables analíticas de valor, como el ratio peso/potencia o la edad del vehículo en el momento de su primer fallo grave.

---

## 9. Riesgos del modelo de datos
* **La parte más clara:** La disponibilidad, estructura estática y calidad de los datos técnicos de Kaggle, así como su ingesta inicial.
* **La mayor incertidumbre:** El porcentaje de éxito real del algoritmo de *Fuzzy Matching*. Si la tasa de coincidencia entre los nombres de Kaggle y los de la API cae por debajo del 40%, el volumen de la capa gold será insuficiente.
* **Plan de Contingencia (Plan B):** Si la fusión con la API de la NHTSA resulta inviable por falta de coincidencias o bloqueos técnicos, el modelo predictivo pivotará. Se prescindirá de los datos de averías y se utilizará exclusivamente la base de datos de Kaggle para predecir la **depreciación del valor de mercado** (precio original MSRP vs. precio actual) en función de las especificaciones mecánicas del vehículo.