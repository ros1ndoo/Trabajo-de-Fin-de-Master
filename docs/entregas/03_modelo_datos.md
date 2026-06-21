# Entrega 3 - Diseño del modelo de datos y capa gold

## 1. Resumen de la idea y datos del proyecto
El proyecto busca resolver el problema de asimetría de información al que se enfrentan los consumidores al adquirir vehículos, basando las decisiones de compra en datos empíricos de fiabilidad en lugar de marketing.

La solución es construir un modelo predictivo (y un dashboard interactivo) que estime la fiabilidad de lanzamientos recientes analizando el historial mecánico de la marca y las especificaciones técnicas del motor. Para garantizar la viabilidad del cruce de datos, **el proyecto se acotará estrictamente al mercado estadounidense para vehículos fabricados desde 1995 hasta la actualidad**. 

Se utilizarán dos fuentes de datos principales y concretas:
* **API de la NHTSA (National Highway Traffic Safety Administration):** Proporcionará los registros oficiales de fallos mecánicos y *recalls* en EE. UU. (A partir de 1995, la NHTSA garantiza un 99% de precisión en la decodificación de datos).
* **Dataset técnico de Kaggle: "Car Features and MSRP" (publicado por CooperUnion):** Aportará la base de datos estática con las especificaciones técnicas exactas (potencia, cilindros, categoría, etc.) filtradas para el mercado norteamericano.

---

## 2. Tecnología o formato de almacenamiento elegido
El ecosistema de datos requerirá un almacenamiento híbrido para procesar diferentes formatos antes de consolidarlos para el Machine Learning:

* **Ficheros JSON y CSV (Capa Raw):** Las peticiones a la API de la NHTSA se descargarán inicialmente en JSON (utilizando peticiones por lotes/batch para evitar bloqueos por *rate limits*), mientras que los datos estáticos de Kaggle se ingerirán en CSV.
* **Base de datos relacional (SQLite):** Se utilizará como motor de procesamiento intermedio. Permitirá realizar cruces complejos (*joins* y agregaciones) mediante SQL sin desbordar la memoria RAM del entorno de desarrollo.
* **Ficheros Parquet (Capa Gold):** Formato final elegido para almacenar el dataset analítico. Se justifica por su compresión eficiente, su estructura columnar que acelera las consultas para dashboards y su capacidad para preservar de forma estricta los tipos de datos en Python/Pandas.

---

## 3. Estructura de capas de datos
El repositorio seguirá una arquitectura de medallón estructurada en los siguientes directorios lógicos:

* `data/raw/`: Datos originales e inmutables. JSONs devueltos por la API de la NHTSA y el CSV técnico de Kaggle.
* `data/processed/`: Datos intermedios donde se aplicarán filtros (ej. exclusión de vehículos anteriores a 1995), imputación de nulos y estandarización de textos.
* `data/gold/`: Dataset final (`.parquet`) que contendrá el cruce exitoso de características técnicas e índices de fallos, listo para entrenar el modelo predictivo.

---

## 4. Definición de la capa gold y la Variable Objetivo
El entregable final de la preparación de datos será una tabla maestra que actuará como contrato de datos para el análisis y modelado.

* **Nombre del dataset:** `gold_us_car_reliability.parquet`
* **Descripción funcional:** Tabla consolidada que vincula las especificaciones mecánicas de cada vehículo con su índice histórico de *recalls* en EE. UU.
* **Nivel de granularidad:** Una fila por marca, modelo, motorización y año.
* **Número esperado de registros:** Entre 30.000 y 80.000 registros.
* **Clave primaria:** Clave compuesta (`id_marca_modelo_motor_ano`).
* **Variable objetivo (`indice_fiabilidad`):** Variable numérica continua (ej. escala de 0 a 100).
* **Justificación y cálculo de la variable objetivo:** Entendiendo que los *recalls* son un proxy legal y no una métrica de fiabilidad mecánica directa, el `indice_fiabilidad` no será un simple conteo bruto. Se calculará como una métrica ponderada que penalice el número de *recalls* **anualizados** (para no castigar injustamente a los coches más antiguos) y ponderados por **gravedad** (usando palabras clave de la descripción del fallo de la NHTSA, penalizando más un fallo de motor/frenos que uno de infoentretenimiento). Esto la convierte en una métrica defendible y estandarizada.

---

## 5. Relaciones entre datos y Auditoría
El mayor desafío técnico es cruzar la base de datos técnica (Kaggle) con los reportes de incidentes (NHTSA).

* **Claves lógicas:** La relación se establecerá mediante la combinación de `Marca` + `Modelo` + `Año`.
* **Tipo de relación (N:M):** Un mismo motor se usa en diferentes modelos, y un mismo modelo experimenta múltiples *recalls* independientes a lo largo del tiempo.
* **Técnica de cruce:** No se utilizarán `JOINs` tradicionales estrictos debido a la inconsistencia en nombres comerciales. Se aplicarán técnicas de **Fuzzy Matching** (ej. librería *TheFuzz*). 
* **Auditoría del cruce:** Para asegurar que el algoritmo no crea "falsos positivos" (ej. unir un Ford F-150 con un Ford F-250), se generará automáticamente un archivo `audit_fuzzy_matches.csv` en la capa *processed*. Este registro capturará las parejas unidas junto con su *Score* de similitud, permitiendo una revisión humana por muestreo (especialmente en aquellos emparejamientos con un *score* límite, entre 80% y 90%) antes de consolidar la capa Gold.

---

## 6. Diccionario de datos inicial

| Campo | Descripción | Tipo de dato | Fuente | Obligatorio | Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id_vehiculo` | Identificador único compuesto | string | Derivado | Sí | Formato: MARCA_MODELO_AÑO |
| `marca` | Fabricante del vehículo | string | Kaggle/NHTSA | Sí | Normalizado a minúsculas |
| `modelo` | Nombre comercial en EE. UU. | string | Kaggle/NHTSA | Sí | Emparejado vía Fuzzy Matching auditado |
| `ano_fabricacion` | Año de lanzamiento | int | Kaggle | Sí | Rango estricto: >= 1995 |
| `cilindros_motor` | Cantidad de cilindros | int | Kaggle | Sí | |
| `potencia_cv` | Potencia (Engine HP) | float | Kaggle | No | Imputación mediante mediana si es nulo |
| `tasa_recalls_anual`| Recalls divididos por los años desde el lanzamiento | float | NHTSA | Sí | Normaliza el sesgo de antigüedad |
| `indice_fiabilidad` | Variable objetivo (Recalls ponderados x gravedad x tiempo)| float | Derivado | Sí | Calculada post-agregación. Proxy validado. |

---

## 7. Problemas de calidad esperados
Al integrar fuentes dispares, se anticipan los siguientes riesgos de calidad que deberán ser mitigados mediante código:

* **Throttling y límites de la API:** Las peticiones continuas a la NHTSA pueden provocar bloqueos de IP, mitigado mediante peticiones *batch* y *delays*.
* **Discrepancia semántica (Nomenclaturas):** Diferencias menores en la forma de escribir los modelos.
* **Evolución del registro de fallos:** Cambios en la taxonomía que usa la NHTSA para definir la "gravedad" de un *recall* entre los años 1995 y 2024.
* **Valores nulos en especificaciones:** Fichas técnicas incompletas en el dataset de Kaggle para vehículos minoritarios.

---

## 8. Decisiones de limpieza y transformación previstas
Se establecerán las siguientes reglas en el *pipeline* de datos hacia la capa processed:

* **Filtro temporal estricto:** Eliminación inmediata de cualquier registro donde `ano_fabricacion < 1995`.
* **Normalización de *Strings*:** Transformación de texto a minúsculas, eliminación de signos de puntuación y aplicación de *Fuzzy Matching* con un umbral estricto del 85%, respaldado por el log de auditoría.
* **Tratamiento de nulos:** Se imputará la mediana de la marca/categoría para variables técnicas continuas faltantes.
* **Feature Engineering:** Creación de variables analíticas (ej. ratio peso/potencia) y el cálculo matemático que dará lugar al `indice_fiabilidad` definitivo.

---

## 9. Riesgos del modelo de datos
* **La parte más clara:** La disponibilidad y estructura del dataset de Kaggle, así como la ingesta inicial desde la API.
* **La mayor incertidumbre:** El porcentaje de éxito real del algoritmo de *Fuzzy Matching*. 
* **Mitigación:** La creación del log de auditoría permitirá ajustar el umbral de similitud empíricamente. Si la tasa de coincidencia cae por debajo del 40% o los falsos positivos son inasumibles, se activará el Plan de Contingencia.
* **Plan de Contingencia:** Se prescindirá de los datos de averías de la API y se utilizará exclusivamente la base de datos de Kaggle para predecir la **depreciación del valor de mercado** (precio original MSRP vs. precio actual) en función de las especificaciones mecánicas del vehículo.
