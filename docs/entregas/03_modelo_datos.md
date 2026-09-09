# Entrega 3 - Diseño del modelo de datos y capa gold (Revisión)

## 1. Resumen de la idea y datos del proyecto
El proyecto busca resolver el problema de asimetría de información al que se enfrentan los consumidores al adquirir vehículos, basando las decisiones de compra en datos empíricos de fiabilidad en lugar de marketing.

La solución es construir un modelo predictivo (y un dashboard interactivo) que estime la **propensión a fallos de seguridad** de lanzamientos recientes analizando el historial de *recalls* de la marca y las especificaciones técnicas. Es fundamental precisar que los *recalls* funcionan como un **proxy legal y observable** de la fiabilidad mecánica, no como una medida directa de desgaste cotidiano: un *recall* de seguridad no equivale necesariamente a una avería mecánica general. Esta distinción se mantendrá visible en el producto final. Para garantizar la viabilidad del cruce de datos, **el proyecto se acotará estrictamente al mercado estadounidense para vehículos fabricados desde 1995 hasta la actualidad**.

Se utilizarán dos fuentes de datos principales y concretas:
* **API de la NHTSA (National Highway Traffic Safety Administration):** Proporcionará los registros oficiales de *recalls* en EE. UU. (A partir de 1995, la NHTSA garantiza un 99% de precisión en la decodificación de datos).
* **Dataset técnico de Kaggle: "Car Features and MSRP" (publicado por CooperUnion):** Aportará la base de datos estática con las especificaciones técnicas exactas (potencia, cilindros, categoría, etc.) filtradas para el mercado norteamericano. Esta fuente queda cerrada como contrato de datos concreto y estable.

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
El entregable final de la preparación de datos será una tabla maestra que actuará como contrato de datos para el análisis y modelado. Para asegurar la consistencia absoluta en el cruce de las fuentes, la granularidad excluye la motorización específica (ya que la NHTSA suele emitir *recalls* a nivel de modelo general) y se establece a nivel de `Marca`, `Modelo` y `Año`. Las características técnicas de las distintas motorizaciones de un mismo modelo-año se agregarán usando la mediana en la capa processed.

* **Nombre del dataset:** `gold_us_car_reliability.parquet`
* **Nivel de granularidad:** Una fila por Marca, Modelo y Año de fabricación.
* **Clave primaria:** Clave compuesta (`id_marca_modelo_ano`).
* **Variable objetivo (`indice_fiabilidad_100`):** Variable numérica continua (escala de 0 a 100, donde 100 es mínima propensión a recalls). Se trata de un **proxy construido**, no una medida directa de fiabilidad mecánica general.
* **Justificación y cálculo de la variable objetivo:** Entendiendo que los *recalls* son un proxy legal y no una métrica de fiabilidad mecánica directa, no se puede usar un conteo bruto. La métrica se calculará en tres fases para asegurar su comparabilidad y defendibilidad:
    1. **Ponderación por gravedad:** Se asignará un peso a cada recall buscando palabras clave en la descripción de la NHTSA.
        * *Crítico (Peso 3.0):* `engine`, `transmission`, `brake`, `steering`, `fire`.
        * *Moderado (Peso 1.5):* `suspension`, `airbag`, `electrical`, `fuel`.
        * *Leve (Peso 1.0):* `infotainment`, `trim`, `paint`, `seat`, `visibility`.
    2. **Ventana de observación común (mitigación del sesgo de antigüedad):** Para hacer comparables un coche de 2005 con uno de 2024, **no se dividirá por la antigüedad total**, ya que un vehículo reciente no ha tenido tiempo suficiente para manifestar sus fallos. En su lugar, se utilizarán únicamente los **recalls emitidos durante los primeros 3 años desde el lanzamiento del modelo** como ventana de observación. Solo los modelos que hayan completado dicha ventana de 3 años participarán en el entrenamiento y evaluación del modelo.
    3. **Mitigación del sesgo de volumen y escalado:** Dado que los coches más vendidos (ej. Ford F-150) estadísticamente sufren más recalls brutos, la tasa ponderada se estandarizará **dentro de su propio segmento de mercado** (ej. comparar Pickups contra Pickups, Sedanes contra Sedanes). La normalización (Z-score) se calculará **exclusivamente sobre el conjunto de entrenamiento** para evitar filtrar información del conjunto de test al conjunto de train. El resultado se escalará a un rango de 0 a 100.

---

## 5. Relaciones entre datos y Auditoría (Fuzzy Matching)
El mayor desafío técnico es cruzar la base de datos técnica (Kaggle) con los reportes de incidentes (NHTSA) debido a la inconsistencia en los nombres comerciales.

* **Claves lógicas:** La relación se establecerá estrictamente mediante la combinación de `Marca` + `Modelo` + `Año`.
* **Técnica de cruce:** Se utilizará **Fuzzy Matching** (función `token_set_ratio` de la librería *TheFuzz*, que es ideal para omitir sufijos de niveles de equipamiento).
* **Auditoría Accionable:** Para evitar falsos positivos al unir los datasets, el *pipeline* aplicará las siguientes reglas y umbrales concretos, y los resultados quedarán registrados en `audit_fuzzy_matches.csv` para inspección:
    * **Automático (Score >= 90):** Match seguro. Se cruza automáticamente.
    * **Rechazo (Score < 75):** Descarte automático.
    * **Revisión Manual (Score 75 - 89):** Estos registros irán al CSV de auditoría. **Criterio de aceptación humana:** Se aceptará el cruce si y solo si el *root name* (nombre principal) coincide y las diferencias se deben únicamente a nomenclaturas de sub-versiones (Ej. Aceptar "Civic" uniendo con "Civic LX", pero rechazar "F-150" uniendo con "F-250").

---

## 6. Diccionario de datos de la Capa Gold

| Campo | Descripción | Tipo de dato | Fuente | Obligatorio | Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id_vehiculo_ano` | Identificador único compuesto | string | Derivado | Sí | PK. Formato: MARCA_MODELO_AÑO |
| `marca` | Fabricante del vehículo | string | Kaggle/NHTSA | Sí | Normalizado a minúsculas |
| `modelo` | Nombre comercial en EE. UU. | string | Kaggle/NHTSA | Sí | Emparejado vía Fuzzy Matching |
| `ano_fabricacion` | Año de lanzamiento | int | Kaggle | Sí | Rango estricto: >= 1995 |
| `categoria_vehiculo`| Segmento del coche (SUV, Sedan, etc.) | string | Kaggle | Sí | Base para aislar el sesgo de ventas |
| `mediana_cilindros` | Mediana de cilindros del modelo/año | int | Kaggle (Agregado)| Sí | Agregado a nivel de Modelo-Año |
| `mediana_cv` | Mediana de potencia del modelo/año| float | Kaggle (Agregado)| Sí | Imputación si es nulo antes de agregar |
| `score_recalls_bruto`| Suma ponderada de gravedad de recalls en ventana de 3 años | float | NHTSA | Sí | Sin normalizar. Solo primeros 3 años desde lanzamiento |
| `indice_fiabilidad_100`| Variable objetivo final (proxy de recalls) | float | Derivado | Sí | Z-score intra-segmento calculado sobre train, escalado 0-100 |

---

## 7. Problemas de calidad esperados
Al integrar fuentes dispares, se anticipan los siguientes riesgos de calidad que deberán ser mitigados mediante código:

* **Throttling y límites de la API:** Las peticiones continuas a la NHTSA pueden provocar bloqueos de IP, mitigado mediante peticiones *batch* y *delays*.
* **Discrepancia semántica (Nomenclaturas):** Diferencias en la forma de escribir los modelos (tratado vía Fuzzy Matching auditado con umbrales explícitos y CSV de revisión).
* **Evolución del registro de fallos:** Cambios en la taxonomía que usa la NHTSA para definir los fallos entre los años 1995 y 2024. El diccionario de palabras clave soluciona esta inconsistencia estandarizando la gravedad.
* **Valores nulos en especificaciones:** Fichas técnicas incompletas en el dataset de Kaggle para vehículos minoritarios.
* **Cohortes incompletas:** Modelos lanzados en los últimos 3 años que todavía no han completado la ventana de observación quedarán excluidos del entrenamiento pero podrán usarse como conjunto de predicción real.

---

## 8. Decisiones de limpieza y transformación previstas
Se establecerán las siguientes reglas en el *pipeline* de datos hacia la capa processed:

* **Agregación Técnica:** Como la capa gold se basa en `Marca-Modelo-Año`, las filas del dataset de Kaggle que contengan distintas versiones de motorización para el mismo modelo-año se agruparán (`groupby`) calculando la mediana de sus atributos (potencia, cilindros) para unificar la granularidad con la NHTSA.
* **Filtro temporal estricto:** Eliminación inmediata de cualquier registro donde `ano_fabricacion < 1995`.
* **Filtro de ventana de observación:** Solo participan en el dataset de entrenamiento los modelos cuya ventana de 3 años desde el lanzamiento haya sido completada en su totalidad.
* **Tratamiento de nulos:** Se imputará la mediana de la marca/categoría para variables técnicas continuas faltantes antes de la agregación.
* **Feature Engineering:** Cálculo de variables relativas y ejecución de la normalización matemática del `indice_fiabilidad_100` **exclusivamente sobre el conjunto de entrenamiento**.

---

## 9. Riesgos del modelo de datos
* **La parte más clara:** La disponibilidad del dataset de Kaggle identificado (CooperUnion) y la estructuración de la ingesta desde la API de la NHTSA.
* **La mayor incertidumbre:** El porcentaje de éxito real del algoritmo de *Fuzzy Matching* tras la revisión manual de los casos dudosos, y la cantidad de modelos que quedarán excluidos por no haber completado la ventana de observación de 3 años.
* **Mitigación y Plan de Contingencia:** Si tras evaluar el archivo `audit_fuzzy_matches.csv`, la tasa de pérdida de modelos (registros huérfanos) supera el 40%, se activará el plan alternativo. Este consiste en prescindir de los datos de *recalls* de la API y utilizar exclusivamente la base de datos de Kaggle para predecir la **depreciación del vehículo** (cruzando el MSRP original de Kaggle con un *scraper* de precios actuales de segunda mano en portales de motor para calcular la pérdida de valor según sus características técnicas).
