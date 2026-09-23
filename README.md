
# AutoReliability — Predicción y análisis de recalls de seguridad

AutoReliability es un proyecto de Data Science desarrollado como Trabajo de Fin de Máster. Su objetivo es reducir la asimetría de información en el mercado automovilístico mediante el análisis de registros históricos de campañas de retirada por defectos de seguridad (*recalls*).

El proyecto integra datos técnicos de vehículos y registros oficiales de la National Highway Traffic Safety Administration (NHTSA) de Estados Unidos para construir un índice estadístico de propensión a recalls.

El resultado se presenta mediante una aplicación web interactiva desarrollada con Streamlit, que permite consultar vehículos, obtener estimaciones históricas, comparar alternativas y examinar sus campañas oficiales de seguridad.

> **Importante:** AutoReliability utiliza los recalls como indicador indirecto de seguridad. Sus resultados no representan una medida de fiabilidad mecánica general, una probabilidad individual de avería ni una certificación de seguridad de un vehículo concreto.

---

## 1. Problema y propuesta de valor

La compra de un vehículo implica una decisión económica importante en la que el comprador dispone de información limitada sobre determinados aspectos de su seguridad.

Aunque es posible consultar el precio, las prestaciones y otras características técnicas, analizar y contextualizar el historial de recalls de un fabricante o modelo puede requerir consultar y comparar numerosos registros.

AutoReliability aborda este problema mediante una herramienta que permite:

- Consultar vehículos por marca, modelo y año.
- Obtener un índice estadístico basado en el historial de recalls.
- Examinar información contextual sobre el fabricante y la categoría del vehículo.
- Comparar los resultados de dos vehículos.
- Consultar campañas oficiales de recalls registradas en la NHTSA.
- Exportar resultados para su posterior consulta.

La herramienta pretende aportar información complementaria durante la investigación previa a la compra de un vehículo.

No sustituye una inspección mecánica, la consulta del historial individual de un vehículo ni las comprobaciones oficiales de recalls pendientes mediante su VIN.

---

## 2. Estado actual y alcance del proyecto

AutoReliability dispone de un prototipo funcional con un pipeline de procesamiento de datos, modelos estadísticos, evaluación temporal y una aplicación web interactiva.

### Cobertura del predictor

| Característica | Estado actual |
|---|---|
| Mercado | Estados Unidos |
| Cobertura técnica del predictor | 1995–2017 |
| Catálogo técnico | 2.213 combinaciones marca-modelo-año |
| Marcas del catálogo | 48 |
| Conjunto de datos Gold | 1.361 registros de vehículos |
| Marcas representadas en Gold | 44 |
| Modelo seleccionado | Baseline histórico por marca y categoría |
| Escala del índice | 0–100 |

La cobertura técnica procede principalmente del conjunto de datos Car Features and MSRP de CooperUnion.

La aplicación también incorpora un catálogo oficial de recalls que permite consultar vehículos posteriores a 2017.

**La disponibilidad de registros oficiales recientes no implica que el modelo predictivo esté entrenado o validado para vehículos posteriores a 2017.**

Se han incorporado fuentes complementarias de EPA/DOE y se han desarrollado herramientas para estudiar la ampliación del catálogo. Sin embargo, estos datos todavía no forman parte del modelo predictivo publicado.

### Estado de desarrollo

El proyecto dispone de:

- Arquitectura de datos Raw → Processed → Gold.
- Ingesta de registros oficiales NHTSA.
- Normalización y cruce de identidades de vehículos.
- Construcción de la variable objetivo.
- Entrenamiento y comparación de diferentes modelos.
- Evaluación temporal y auditoría de errores por grupos.
- Aplicación web con consultas, comparaciones y exportaciones.
- Pruebas automatizadas e integración continua.

El sistema debe considerarse un prototipo funcional de análisis y estimación histórica, no un predictor universal validado para todos los vehículos del mercado.

---

## 3. Fuentes de datos

### 3.1. NHTSA — Recalls oficiales

La National Highway Traffic Safety Administration es la fuente utilizada para obtener información sobre campañas de retirada por defectos de seguridad.

Se utilizan registros oficiales que contienen información como:

- Fabricante, modelo y año del vehículo.
- Identificación de la campaña.
- Fecha de comunicación.
- Componente afectado.
- Descripción del defecto.
- Consecuencias potenciales y medidas correctivas disponibles.

El proyecto incorpora una descarga masiva de registros oficiales y permite realizar consultas bajo demanda.

Los registros utilizados para construir la variable objetivo y los consultados posteriormente por un usuario se mantienen separados.

Fuente: https://www.nhtsa.gov/recalls

### 3.2. CooperUnion — Car Features and MSRP

El conjunto de datos Car Features and MSRP proporciona las especificaciones técnicas utilizadas para construir el catálogo original del predictor.

Entre los campos utilizados se encuentran:

- Marca.
- Modelo.
- Año.
- Categoría o carrocería.
- Potencia.
- Número de cilindros.

Las diferentes versiones técnicas de un mismo modelo-año se agregan para construir una representación común.

El sistema conserva la procedencia de las especificaciones y transforma las unidades cuando corresponde.

Fuente: https://www.kaggle.com/datasets/CooperUnion/cardataset

### 3.3. EPA/DOE — Fuentes complementarias

El proyecto incorpora herramientas para adquirir y procesar información adicional procedente de EPA/DOE, incluyendo inventarios de vehículos y registros de vehículos utilizados en pruebas de consumo.

Estos datos permiten investigar la ampliación de la cobertura temporal y mejorar la identificación de vehículos.

Actualmente permanecen separados del conjunto de entrenamiento y del modelo publicado.

Su integración requiere verificar las equivalencias entre vehículos, la disponibilidad de las características necesarias y la validez de una nueva evaluación temporal.

---

## 4. Arquitectura de datos

El pipeline sigue una arquitectura de tres capas:

### Raw

Contiene las respuestas originales y los archivos descargados de las fuentes de información.

Se conservan los datos originales, los metadatos de procedencia y, cuando corresponde, los hashes utilizados para comprobar su integridad.

### Processed

Contiene las transformaciones intermedias necesarias para preparar los datos:

- Normalización de nombres y unidades.
- Limpieza de especificaciones técnicas.
- Identificación y cruce de vehículos.
- Tratamiento de registros de recalls.
- Auditoría de coincidencias.
- Preparación de la información temporal.

Se utiliza SQLite como almacenamiento intermedio para determinadas operaciones.

### Gold

Contiene el conjunto de datos preparado para el entrenamiento y evaluación de los modelos.

Cada registro representa una combinación de marca, modelo y año admitida por el proceso de construcción de datos.

El dataset incluye las características predictoras, la información temporal y la variable objetivo correspondiente.

La ausencia de una campaña en una fuente incompleta no debe interpretarse automáticamente como un vehículo con cero recalls.

---

## 5. Construcción de la variable objetivo

El proyecto utiliza una variable denominada `indice_fiabilidad_100`.

Aunque conserva este nombre interno, debe interpretarse como un índice relativo basado en recalls de seguridad y no como una medición de la fiabilidad mecánica general.

### 5.1. Ventana temporal

Para construir la variable objetivo se consideran las campañas registradas durante los tres primeros años-modelo de observación del vehículo.

Por ejemplo, para un vehículo de 2012 se consideran las campañas cuya fecha de reporte corresponde a 2012, 2013 o 2014.

Esta ventana permite establecer un periodo común de observación para las diferentes cohortes.

### 5.2. Ponderación de campañas

Las campañas se clasifican mediante reglas basadas en palabras clave presentes en sus descripciones.

Se utilizan tres categorías:

| Categoría | Peso |
|---|---:|
| Crítica | 3,0 |
| Moderada | 1,5 |
| Baja o no clasificada | 1,0 |

Los pesos se suman para obtener una puntuación bruta de recalls.

Esta clasificación es una aproximación heurística definida para el proyecto. No representa una escala oficial de gravedad de la NHTSA ni una medición validada de las consecuencias reales de cada campaña.

### 5.3. Normalización

La puntuación bruta se transforma en un índice de 0 a 100 utilizando estadísticas de referencia calculadas exclusivamente con los datos de entrenamiento.

La fórmula general es:

\[
I = \operatorname{clip}
\left(
50 - 10 \cdot \frac{R-\mu_s}{\sigma_s},
0,
100
\right)
\]

Donde:

- \(I\): índice normalizado.
- \(R\): puntuación bruta ponderada de recalls.
- \(\mu_s\): media de la puntuación bruta del segmento de referencia.
- \(\sigma_s\): desviación estándar correspondiente.
- `clip`: limita el resultado al intervalo de 0 a 100.

Cuando no existe una referencia válida para el segmento, se aplica la política de referencia definida en el normalizador del entrenamiento.

### Interpretación del índice

**Cuanto mayor es la puntuación, menor es la carga histórica estimada de recalls en la escala utilizada.**

| Índice | Interpretación |
|---|---|
| 100 | Extremo de menor carga de recalls |
| 50 | Referencia central de la normalización |
| 0 | Extremo de mayor carga de recalls |

La escala es relativa a las estadísticas de referencia utilizadas.

Una puntuación de 80 no significa que el vehículo tenga un 80 % de fiabilidad ni que exista un 20 % de probabilidad de sufrir una avería.

Del mismo modo, una diferencia de diez puntos entre dos vehículos no equivale necesariamente a una diferencia proporcional en su riesgo real de seguridad.

---

## 6. Modelado predictivo

El proyecto implementa y evalúa tres alternativas.

### Baseline histórico

Utiliza las medias del índice observadas en el conjunto de entrenamiento para obtener una referencia estadística.

La estimación se calcula utilizando la combinación de marca y categoría del vehículo.

Cuando no existe información suficiente para esa combinación, el estimador dispone de referencias más generales, sujetas a las restricciones de disponibilidad establecidas por el servicio.

El baseline no utiliza la potencia, los cilindros ni el historial reciente de marca para diferenciar las estimaciones.

Por tanto, dos vehículos pertenecientes a la misma combinación de marca y categoría pueden recibir la misma puntuación, aunque correspondan a modelos o años diferentes.

### Ridge Regression

Modelo de regresión lineal regularizada que permite utilizar características numéricas y categóricas del vehículo.

### Random Forest

Modelo de ensamblado basado en árboles de decisión que permite capturar relaciones no lineales entre las características disponibles.

Los tres candidatos se comparan mediante un protocolo de validación temporal.

El criterio de selección incorpora tanto el error de validación como una condición de mejora mínima para el modelo avanzado frente a Ridge.

### Modelo seleccionado

El artefacto publicado actualmente utiliza el baseline histórico.

En la evaluación registrada, el baseline obtuvo el menor MAE de validación entre los modelos elegibles.

Random Forest presentó un resultado muy próximo, pero no cumplió el umbral de mejora requerido frente a Ridge para ser seleccionado.

Por tanto, el modelo final proporciona principalmente una estimación histórica por marca y categoría, no una predicción individualizada basada en todas las características técnicas del vehículo.

---

## 7. Evaluación del modelo

El entrenamiento y la evaluación utilizan particiones temporales y un embargo que tiene en cuenta la madurez de la ventana de observación.

Esta estrategia pretende evitar que el modelo utilice resultados futuros que no habrían estado disponibles en el periodo correspondiente.

### Resultados de validación

| Modelo | MAE |
|---|---:|
| Baseline histórico | 10,7412 |
| Ridge Regression | 11,3099 |
| Random Forest | 10,7460 |

La validación registrada incluye 90 observaciones.

### Resultados de test

| Métrica | Resultado |
|---|---:|
| Número de observaciones | 819 |
| Periodo evaluado | 2012–2017 |
| MAE | 12,1984 |
| RMSE | 17,0354 |

El MAE representa la diferencia absoluta media entre las puntuaciones predichas y las puntuaciones objetivo observadas.

Un MAE de 12,1984 significa que el error absoluto medio en la muestra evaluada fue de aproximadamente 12,2 puntos sobre la escala del índice.

No significa que todas las predicciones estén dentro de un intervalo de ±12,2 puntos.

### Limitaciones de la evaluación

Las métricas describen el comportamiento del modelo en la población y los periodos evaluados.

No garantizan el mismo rendimiento para cualquier marca, modelo, categoría o año.

El análisis por grupos ha identificado diferencias de error entre fabricantes y segmentos. Por tanto, el MAE global no debe interpretarse como una garantía de precisión individual.

El conjunto de test registrado ya ha sido examinado y no debe considerarse una muestra independiente intacta para optimizaciones posteriores.

Además, el modelo final no ha demostrado superar al baseline con un error homogéneo entre grupos, como se planteaba en los objetivos metodológicos originales.

Los resultados deben interpretarse dentro de estas limitaciones.

Los experimentos y sus métricas se encuentran en `artifacts/`, mientras que el código de evaluación y auditoría está disponible en `src/auto_reliability/`.

---

## 8. Aplicación web

La interfaz está desarrollada con Streamlit y permite acceder a las funcionalidades del proyecto mediante dos apartados principales.

### 8.1. Predicción de propensión a recalls

El usuario selecciona una marca, un modelo y un año del catálogo técnico disponible.

La aplicación muestra:

- Índice histórico estimado.
- Modelo o referencia utilizada.
- MAE global, cuando corresponde al artefacto evaluado.
- Historial previo disponible.
- Características técnicas y contexto del segmento.
- Explicación de los factores utilizados por el estimador.
- Procedencia y limitaciones del resultado.

Cuando el modelo seleccionado es el baseline, la explicación identifica la referencia de marca y categoría utilizada.

No atribuye efectos predictivos individuales a características que el baseline no utiliza.

La aplicación también permite añadir dos resultados a una comparación y exportar información en CSV o PDF.

### 8.2. Recalls oficiales NHTSA

El segundo apartado permite seleccionar un vehículo y consultar las campañas oficiales disponibles para su marca, modelo y año.

Esta funcionalidad utiliza un catálogo oficial independiente del catálogo técnico del predictor.

Por tanto, puede ofrecer consultas de vehículos más recientes que los incluidos en el entrenamiento.

Los recalls obtenidos mediante esta consulta son información observada a la fecha de recuperación.

**No se incorporan como variables predictoras del propio vehículo ni provocan un reentrenamiento automático del modelo.**

### 8.3. Explicaciones generativas

El proyecto dispone de una integración opcional con un modelo de lenguaje local mediante Ollama.

La integración está diseñada para trabajar con información procedente del estimador y limitar las afirmaciones generadas a los factores disponibles.

Cuando no existe un modelo local configurado, el sistema utiliza explicaciones deterministas.

La integración generativa no interviene en el cálculo de las predicciones.

No se ha acreditado una evaluación completa de esta integración con un proveedor real dentro de los resultados de validación publicados.

---

## 9. Tecnologías utilizadas

| Área | Tecnologías |
|---|---|
| Lenguaje principal | Python |
| Procesamiento de datos | Pandas, NumPy |
| Almacenamiento | SQLite, Parquet |
| Adquisición de datos | Requests |
| Cruce de identidades | TheFuzz |
| Machine Learning | Scikit-learn |
| Serialización del modelo | Joblib |
| Aplicación web | Streamlit |
| Visualización | Plotly, Matplotlib |
| Generación de informes | ReportLab |
| Pruebas | Pytest |
| Análisis estático | Ruff |
| Integración continua | GitHub Actions |

El proyecto incorpora además utilidades para trabajar con archivos XML, XLSX y snapshots de las fuentes complementarias.

---

## 10. Instalación y ejecución

### Requisitos

- Python 3.10 o superior.
- Git.
- Acceso a Internet para descargar dependencias o realizar consultas externas.
- PowerShell en Windows para utilizar los scripts de instalación y arranque incluidos.

El repositorio contiene el modelo entrenado y los archivos Gold necesarios para cargar el predictor publicado.

No es necesario volver a entrenar el modelo para iniciar la aplicación con los artefactos existentes.

Algunas funcionalidades de consulta oficial requieren acceso a la NHTSA o disponer de los datos oficiales descargados localmente.

### 10.1. Clonar el repositorio

```bash
git clone https://github.com/ros1ndoo/Trabajo-de-Fin-de-Master.git

cd Trabajo-de-Fin-de-Master
```

### 10.2. Preparar el entorno en Windows

Desde PowerShell, ejecutar:

```powershell
.\scripts\setup.ps1
```

Este script crea el entorno virtual e instala las dependencias bloqueadas del proyecto.

La preparación inicial puede requerir tiempo y conexión a Internet.

### 10.3. Iniciar AutoReliability

```powershell
.\scripts\start.ps1
```

La aplicación se inicia localmente y queda disponible en:

http://127.0.0.1:8501

El script de arranque utiliza el entorno virtual previamente preparado y no reinstala las dependencias en cada ejecución.

### 10.4. Instalación manual

Como alternativa, se puede preparar un entorno virtual e instalar el proyecto mediante pip:

```bash
python -m venv .venv
```

Una vez activado el entorno:

```bash
python -m pip install -e .
```

Para iniciar la aplicación:

```bash
python -m streamlit run app.py
```

La instalación manual mediante las dependencias declaradas en `pyproject.toml` puede resolver versiones distintas de las utilizadas durante el desarrollo.

Para reproducir el entorno de Windows con las versiones bloqueadas, debe utilizarse el procedimiento de instalación correspondiente a `requirements.lock`.

---

## 11. Procesamiento de datos y entrenamiento

El proyecto incorpora una interfaz de línea de comandos que permite ejecutar las distintas etapas del pipeline.

Los siguientes comandos se ejecutan desde la raíz del repositorio, con el entorno Python del proyecto activado.

### Consultar el estado de los artefactos

```bash
python -m auto_reliability.cli status
```

### Descargar los datos técnicos originales

```bash
python -m auto_reliability.cli download-data
```

Descarga el conjunto de datos CooperUnion y registra su procedencia.

La operación requiere acceso a la fuente original.

### Descargar e indexar recalls oficiales

```bash
python -m auto_reliability.cli download-recalls
```

Obtiene los archivos masivos oficiales utilizados por el pipeline y construye el índice local.

### Ejecutar el pipeline

```bash
python -m auto_reliability.cli pipeline \
  --technical-csv data/raw/cooperunion_car_features.csv \
  --recall-source bulk \
  --train-end-year 2010
```

El comando anterior utiliza la sintaxis de continuación de líneas de Bash. En PowerShell puede escribirse en una sola línea.

El pipeline construye los datos procesados, realiza el cruce de identidades y genera el conjunto de datos Gold.

### Entrenar y evaluar modelos

Para utilizar los cortes temporales de la ejecución publicada:

```bash
python -m auto_reliability.cli train \
  --train-end-year 2007 \
  --validation-end-year 2011
```

Los parámetros anteriores son importantes: los valores predeterminados del CLI no corresponden a los cortes utilizados por el artefacto publicado.

Una nueva descarga de datos oficiales puede contener actualizaciones y producir resultados diferentes de los obtenidos durante el entrenamiento original.

La reproducción exacta requiere utilizar los mismos snapshots, reglas de procesamiento, versiones de dependencias y parámetros registrados en los artefactos del experimento.

### Auditar el modelo

```bash
python -m auto_reliability.cli audit-model
```

Genera un análisis descriptivo del error del modelo publicado, incluyendo resultados por diferentes grupos.

### Evaluación temporal adicional

```bash
python -m auto_reliability.cli evaluate-temporal
```

Ejecuta los experimentos retrospectivos definidos en el proyecto.

Estos experimentos no sustituyen automáticamente el modelo publicado ni constituyen una nueva evaluación independiente sobre datos que nunca hayan sido examinados.

---

## 12. Estructura del repositorio

```text
Trabajo-de-Fin-de-Master/
│
├── .github/
│   └── workflows/             # Integración continua
│
├── .streamlit/                 # Configuración de la interfaz
│
├── artifacts/
│   ├── reliability_model.joblib
│   ├── model_metrics.json
│   ├── validation_predictions.csv
│   └── test_predictions.csv
│
├── data/
│   ├── raw/                   # Fuentes originales y procedencia
│   ├── processed/             # Transformaciones y almacenamiento intermedio
│   └── gold/                  # Datasets para entrenamiento e inferencia
│
├── docs/
│   ├── entregas/              # Documentación académica original
│   └── assets/                # Recursos de diseño
│
├── scripts/
│   ├── setup.ps1
│   └── start.ps1
│
├── src/
│   └── auto_reliability/
│       ├── acquisition.py     # Descarga de datos
│       ├── bulk_recalls.py    # Ingesta masiva NHTSA
│       ├── data_sources.py    # Fuentes y normalización técnica
│       ├── matching.py        # Cruce de identidades
│       ├── transform.py       # Transformaciones y variable objetivo
│       ├── pipeline.py        # Orquestación Raw → Processed → Gold
│       ├── modeling.py        # Entrenamiento y selección de modelos
│       ├── model_audit.py     # Auditoría de resultados
│       ├── service.py         # Servicio de predicción y consultas
│       ├── explainability.py  # Explicaciones del modelo
│       ├── narrative.py       # Explicación generativa opcional
│       ├── reporting.py       # Generación de informes
│       ├── dashboard.py       # Interfaz Streamlit
│       └── cli.py             # Interfaz de línea de comandos
│
├── tests/                     # Pruebas automatizadas
├── app.py                     # Punto de entrada de la aplicación
├── pyproject.toml             # Configuración del proyecto
├── requirements.txt           # Dependencias declaradas
└── requirements.lock          # Dependencias bloqueadas
```

Los archivos originales de mayor tamaño y determinados artefactos intermedios no están incluidos íntegramente en el repositorio.

Para reconstruir todo el pipeline es necesario disponer de las fuentes correspondientes o descargarlas mediante los comandos previstos.

---

## 13. Pruebas e integración continua

El proyecto dispone de pruebas automatizadas para diferentes componentes:

- Validación de datos y contratos.
- Procesamiento y cruce de identidades.
- Construcción de la variable objetivo.
- Entrenamiento y selección del modelo.
- Comprobaciones temporales.
- Consultas bajo demanda.
- Tratamiento de errores de fuentes externas.
- Interfaz, comparaciones y generación de informes.

Para ejecutar las pruebas:

```bash
python -m pytest -q
```

Para ejecutar el análisis estático:

```bash
python -m ruff check src tests app.py
```

GitHub Actions ejecuta estas comprobaciones en Python 3.10 y Python 3.12.

La ejecución asociada al commit `530ac93` finalizó correctamente en ambos entornos.

[Consultar GitHub Actions](https://github.com/ros1ndoo/Trabajo-de-Fin-de-Master/actions)

Las pruebas automatizadas verifican el comportamiento del software, pero no demuestran por sí solas que las predicciones sean estadísticamente válidas para cualquier vehículo.

---

## 14. Limitaciones y trabajo futuro

AutoReliability es un prototipo académico funcional. Existen limitaciones que deben tenerse en cuenta al interpretar sus resultados.

### Cobertura temporal

El predictor publicado utiliza especificaciones técnicas de vehículos comprendidos entre 1995 y 2017.

La ampliación a lanzamientos actuales requiere incorporar nuevas fuentes técnicas, comprobar su compatibilidad y realizar una nueva evaluación.

### Representatividad de los datos

El conjunto de entrenamiento depende de los vehículos cuyas identidades han podido cruzarse con los registros disponibles.

Persisten casos no resueltos y posibles sesgos de selección.

La ausencia de un registro en un fichero de campañas no acredita por sí sola que un vehículo haya tenido cero recalls.

### Capacidad de diferenciación

El baseline seleccionado proporciona una referencia histórica por marca y categoría.

No distingue necesariamente entre modelos o años diferentes que pertenezcan al mismo grupo.

Tampoco utiliza todas las características técnicas disponibles para individualizar las predicciones.

### Interpretación de los recalls

El índice se construye a partir de campañas oficiales y de una ponderación heurística de sus descripciones.

No incorpora directamente todas las averías mecánicas, la exposición individual al riesgo, el número de unidades afectadas o el estado de reparación de cada vehículo.

Por tanto, no debe utilizarse de forma aislada para decidir qué vehículo es más seguro o fiable.

### Validación predictiva

El modelo publicado no ha demostrado superar al baseline con un error homogéneo entre grupos.

Los resultados de evaluación deben interpretarse dentro de la población histórica utilizada.

Una ampliación del proyecto requeriría mejorar la cobertura de los datos y realizar nuevas evaluaciones temporales, incluyendo el análisis de errores por marca, categoría y año.

### Reproducibilidad y despliegue

El repositorio incluye el código, el modelo publicado, los conjuntos Gold y archivos de evaluación.

La reconstrucción íntegra del procesamiento original depende de fuentes y snapshots que no están incluidos completamente en GitHub.

El despliegue actual está orientado al uso local. La utilización del producto en un entorno de producción requeriría controles adicionales de disponibilidad, seguridad, mantenimiento y monitorización.

---

## 15. Conclusión

AutoReliability demuestra la integración de diferentes etapas de un proyecto de ciencia de datos: adquisición de información, procesamiento, construcción de variables, entrenamiento, evaluación y presentación de resultados mediante una aplicación interactiva.

El proyecto transforma registros históricos de recalls de seguridad en una referencia estadística que puede consultarse y compararse desde una interfaz accesible.

Los resultados actuales permiten ofrecer información histórica complementaria, pero no constituyen una evaluación individual de la fiabilidad mecánica de cada vehículo.

El desarrollo futuro se centraría en ampliar la cobertura de datos, mejorar la representatividad de la población utilizada y evaluar si modelos más individualizados pueden aportar mejoras predictivas demostrables frente a la referencia histórica.

---

**AutoReliability — Trabajo de Fin de Máster en Data Science.**
