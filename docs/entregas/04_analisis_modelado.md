# Entrega 4 - Diseño del análisis y estrategia de modelado

## 1. Problema que se busca resolver
El proyecto aborda la asimetría de información que sufren los consumidores y las empresas en el sector automotriz al adquirir vehículos de reciente lanzamiento. Actualmente, las decisiones de compra se basan casi exclusivamente en el marketing, ya que la fiabilidad real de una motorización solo se conoce empíricamente tras años de uso en el mercado. 

*   **Qué ocurre actualmente y por qué supone un problema:** Se asumen grandes riesgos económicos al comprar vehículos o renovar flotas sin información objetiva sobre la propensión a fallos de los nuevos motores, lo que deriva en gastos imprevistos y tiempos de inactividad.
*   **Quién utilizará el resultado y para qué decisión:** Consumidores particulares y gestores de flotas que necesiten una herramienta analítica de apoyo y recomendación antes de realizar una inversión automotriz.
*   **Qué resultado concreto debería producir el proyecto:** El proyecto se considerará útil si logra generar una estimación anticipada de fiabilidad basada en el historial empírico del fabricante y las especificaciones técnicas, aportando transparencia al mercado.

## 2. Análisis de datos planteado y utilidad esperada
El Análisis Exploratorio de Datos (EDA) se realizará directamente sobre la capa gold para entender las dinámicas de los fallos mecánicos antes de entrenar cualquier algoritmo.

*   **Qué preguntas queréis responder con los datos:** ¿Existe correlación entre motores de alta cilindrada o potencia y una mayor tasa de fallos severos? ¿Qué marcas han mejorado o empeorado sistemáticamente su fiabilidad desde 1995?
*   **Qué análisis vais a estudiar:** Se segmentará la distribución de la variable objetivo agrupándola por categoría de vehículo (SUV, sedán, pickup, etc.) y marca para identificar varianzas, así como la evolución de la fiabilidad a lo largo del tiempo (1995-actualidad).
*   **Qué hipótesis queréis comprobar:** Se busca contrastar la hipótesis de si los vehículos modernos presentan tasas de incidentes distintas a los modelos antiguos debido a la mayor carga electrónica u otros factores.
*   **Qué visualizaciones podrían incorporarse al MVP:** El análisis derivará en visualizaciones, como gráficos interactivos de radar, que compararán las características técnicas del vehículo seleccionado con la media histórica de su categoría.
*   **Cómo ayudará este análisis:** Ayudará a contextualizar el *score* predictivo de fiabilidad para el usuario final y a validar la calidad de las variables seleccionadas para el modelo.

## 3. Tipo de modelos que se van a plantear
El proyecto abordará una tarea analítica de **regresión**, puesto que el propósito es predecir una variable continua numérica (el índice de fiabilidad acotado matemáticamente en una escala de 0 a 100).

| Alternativa | Tipo | Por qué se plantea | Limitación principal |
| :--- | :--- | :--- | :--- |
| **Baseline** | Regla simple (Media condicional) | Proporciona una referencia mínima basándose en la media histórica de fiabilidad de la marca y de su segmento en los últimos años para saber si el modelo aporta mejora. | Ignora por completo las innovaciones o especificaciones técnicas concretas de la nueva motorización. |
| **Modelo candidato 1** | Modelo interpretable (Regresión Lineal Regularizada - Ridge) | Permite construir una primera solución reproducible y explicable mediante los pesos de los coeficientes. | Puede no capturar relaciones complejas o interacciones conjuntas no lineales entre variables (ej. cilindros, categoría y peso). |
| **Modelo candidato 2** | Modelo avanzado (Gradient Boosting - XGBoost / Random Forest) | Permite comprobar si una mayor complejidad captura mejor las interacciones entre las características técnicas y el fabricante para mejorar el resultado. | Mayor riesgo de sobreajuste y menor interpretabilidad nativa, requiriendo SHAP para explicar las decisiones
. |

## 4. Datos de entrada del análisis y los modelos
El modelo consumirá en exclusiva la tabla analítica final de la capa gold. Es imperativo asegurar que no se utilicen datos de reportes de fallos generados a posteriori de la salida al mercado del coche para evitar el riesgo de fuga de información.

| Entrada | Descripción | Granularidad / tipo | Uso en el análisis o modelo |
| :--- | :--- | :--- | :--- |
| `gold_us_car_reliability` | Dataset final procedente de la entrega 3 cruzando fuentes de Kaggle y NHTSA. | Una fila por Marca, Modelo y Año de fabricación. | Fuente principal de variables de entrada. |
| `marca` | Fabricante del vehículo (ej. Toyota, Ford). | Texto (Categórica). | Feature directa para el análisis (se codificará numéricamente para el modelo). |
| `categoria_vehiculo` | Segmento del automóvil (SUV, Compacto, etc.). | Texto (Categórica). | Feature directa para contextualizar el segmento de mercado. |
| `mediana_cilindros` y `mediana_cv` | Especificaciones mecánicas agregadas a nivel de modelo-año. | Numérica. | Variables predictoras de entrada para evaluar la complejidad del motor. |
| `hist_fiabilidad_marca` | Variable calculada a partir de datos históricos: Media móvil de la fiabilidad de la marca en los 3 años previos. | Numérica / Variable derivada. | Resume comportamiento previo y da contexto temporal al modelo sobre los estándares de calidad del fabricante. |

## 5. Datos de salida y forma de consumo
El modelo producirá un resultado predictivo que se consumirá a través del *dashboard* interactivo definido como el MVP del proyecto final. 

| Campo de salida | Descripción | Tipo | Uso posterior |
| :--- | :--- | :--- | :--- |
| `id_vehiculo_ano` | Identificador de la unidad predicha (MARCA_MODELO_AÑO). | string. | Trazabilidad y unión con el resto del proyecto visual en el dashboard. |
| `prediccion_indice_100` | Resultado principal del modelo: estimación de fiabilidad (0 a 100). | float | Dashboard web y recomendación de compra para el usuario. |
| `fecha_ejecucion` | Momento en que se genera el resultado predictivo. | datetime | Control de actualización y reproducibilidad. |
| `explicacion_factores` | Atribución de variables clave que impulsan el resultado predictivo (ej. influencia de la marca o potencia). | texto | Interpretación por parte del usuario para entender los motivos del score. |

## 6. Estrategia para diseñar y seleccionar el modelo
El proceso de selección del modelo seguirá estos pasos para asegurar la utilidad real por encima de la mera optimización de la calidad predictiva:

1.  **Preprocesamiento:** Se imputarán los valores nulos residuales de las especificaciones técnicas utilizando la mediana intra-segmento de la marca. Las variables categóricas se transformarán mediante *One-Hot Encoding* o *Target Encoding*.
2.  **Construcción del baseline:** Generación del modelo de referencia basado en la media histórica.
3.  **Comparación:** Se evaluarán los modelos candidatos (Ridge vs. XGBoost/Random Forest) valorando no solo el error predictivo, sino también su estabilidad, coste computacional y, sobre todo, su interpretabilidad.
4.  **Regla de decisión final:** El modelo más complejo (ensamblado de árboles) solo será seleccionado si logra reducir el error de la regresión lineal simple en un umbral significativo (ej. >10%). Si el rendimiento predictivo es similar, las condiciones para ser seleccionado favorecerán a la regresión lineal por su utilidad explicativa en el MVP.

## 7. Estrategia de validación y evaluación
Dado que el proyecto consiste en proyectar la fiabilidad de lanzamientos futuros basándose en historiales pasados, se evitará la separación aleatoria estándar.

| Elemento | Decisión prevista | Justificación |
| :--- | :--- | :--- |
| **Separación de datos** | Split temporal estricto (Ej: Train 1995-2018, Valid 2019-2021, Test 2022-2024). | Debe evitar la contaminación de datos (leakage) y parecerse al uso real: predecir coches nuevos con datos anteriores. |
| **Métrica principal** | MAE (Mean Absolute Error) y RMSE. | El MAE refleja fielmente la magnitud del error en la escala 0-100 para el usuario final, penalizando los errores graves con RMSE. |
| **Baseline** | Predicción estacionaria (media de la marca en su segmento). | Permite medir la mejora real aportada por las técnicas de Machine Learning. |
| **Criterio de aceptación** | Superar la precisión del baseline logrando una distribución del error homogénea. | Define cuándo el resultado es suficientemente útil y aporta transparencia frente a la intuición. |

## 8. Riesgos y alternativas
La estrategia de modelado plantea las siguientes incertidumbres y vías de mitigación:

*   **Riesgo sobre la variable objetivo:** La variable predictiva se basa en los registros oficiales de *recalls* de la NHTSA. Existe el riesgo de que representen obligaciones legales de seguridad en lugar del desgaste mecánico del día a día, no reflejando la idea coloquial de "fiabilidad".
*   **Riesgo de Data Leakage:** Supone el riesgo más crítico. Se garantiza que atributos como el recuento final de fallos de la API no entren al modelo durante el entrenamiento, limitando las entradas a la técnica disponible el año del lanzamiento.
*   **Calidad y volumen tras cruces:** El algoritmo de *Fuzzy Matching* para cruzar nomenclaturas de Kaggle y la NHTSA puede generar registros huérfanos, desbalanceando los datos o perdiendo marcas enteras. 
*   **Alternativa / Plan de Contingencia:** ¿Qué alternativa se aplicaría si el modelo no supera el baseline o pierde rigor? Si la tasa de pérdida de modelos supera el 40% en el cruce de datos, el proyecto prescindirá de la variable de averías. La alternativa directa será utilizar el dataset técnico para predecir la **depreciación económica o pérdida de valor** del vehículo cruzando precios originales (MSRP) con el mercado de segunda mano.
