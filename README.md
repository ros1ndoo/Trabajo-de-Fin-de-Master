
# AutoReliability — Predicción y análisis de recalls de seguridad

AutoReliability es un proyecto de Data Science desarrollado como Trabajo de Fin de Máster. Su objetivo es reducir la asimetría de información en la compra de vehículos mediante el análisis de registros históricos de campañas de retirada por defectos de seguridad (*recalls*).

El proyecto combina datos técnicos de vehículos con registros oficiales de la **National Highway Traffic Safety Administration (NHTSA)** para generar un índice estadístico y facilitar la consulta y comparación de vehículos mediante una aplicación web interactiva.

> **Importante:** El índice es un indicador indirecto basado en recalls de seguridad. No representa la fiabilidad mecánica general ni la probabilidad individual de sufrir una avería.

## 1. Funcionalidades

La aplicación, desarrollada con Streamlit, permite:

- Consultar vehículos por marca, modelo y año.
- Obtener un índice histórico de propensión a recalls.
- Visualizar características técnicas, historial del fabricante y factores explicativos.
- Comparar los resultados de dos vehículos.
- Consultar campañas oficiales de recalls de la NHTSA.
- Exportar los resultados en CSV y PDF.

![Mockup del frontal](docs/assets/05_mockup_frontal.png)

## 2. Fuentes y procesamiento de datos

El proyecto utiliza las siguientes fuentes:

- **NHTSA:** registros oficiales de campañas de retirada por defectos de seguridad.
- **CooperUnion — Car Features and MSRP:** especificaciones técnicas de vehículos del mercado estadounidense.
- **EPA/DOE:** fuentes complementarias utilizadas para investigar la ampliación de la cobertura temporal.

Los datos se procesan mediante una arquitectura **Raw → Processed → Gold**, que incluye limpieza, normalización, cruce de identidades y construcción del conjunto de datos utilizado para entrenar los modelos.

El predictor publicado utiliza datos técnicos de vehículos comprendidos entre 1995 y 2017. La consulta independiente de recalls oficiales permite acceder a registros de vehículos más recientes.

## 3. Modelo predictivo

### Construcción del índice

La variable objetivo se construye a partir de las campañas de recalls registradas durante los tres primeros años-modelo de cada vehículo.

Las campañas reciben una ponderación heurística según las palabras clave de sus descripciones:

| Categoría | Peso |
|---|---:|
| Crítica | 3,0 |
| Moderada | 1,5 |
| Baja o no clasificada | 1,0 |

La suma ponderada se normaliza respecto a las estadísticas del segmento, calculadas sobre los datos de entrenamiento, para obtener un índice de 0 a 100.

**Una puntuación mayor indica una menor carga histórica estimada de recalls.** No representa un porcentaje de fiabilidad ni una probabilidad individual de avería.

### Entrenamiento y selección

Se han evaluado tres alternativas:

- Baseline histórico por marca y categoría.
- Ridge Regression.
- Random Forest.

El modelo seleccionado actualmente es el **baseline histórico**, que utiliza la media del índice de los vehículos de referencia pertenecientes a una misma marca y categoría.

Por tanto, dos vehículos del mismo grupo pueden recibir la misma estimación aunque tengan diferentes características técnicas o años de fabricación.

### Resultados de evaluación

| Métrica | Resultado |
|---|---:|
| Vehículos de entrenamiento | 290 |
| Vehículos de validación | 90 |
| Vehículos de test | 819 |
| MAE de test | 12,1984 |
| RMSE de test | 17,0354 |

La evaluación utiliza una separación temporal de los datos. Las métricas corresponden a la población histórica evaluada y no garantizan el mismo rendimiento para cualquier vehículo.

El modelo seleccionado no ha demostrado superar al baseline con un error homogéneo entre grupos, uno de los objetivos metodológicos originales del proyecto.

## 4. Tecnologías utilizadas

| Área | Tecnologías |
|---|---|
| Lenguaje | Python |
| Procesamiento de datos | Pandas, NumPy |
| Almacenamiento | SQLite, Parquet |
| Machine Learning | Scikit-learn |
| Visualización | Streamlit, Plotly, Matplotlib |
| Informes | ReportLab |
| Pruebas | Pytest, Ruff, GitHub Actions |

El proyecto también incorpora una integración opcional con Ollama para generar explicaciones a partir de información proporcionada por el modelo. Cuando no está configurada, utiliza explicaciones deterministas.

## 5. Instalación y ejecución

### Requisitos

- Python 3.10 o superior.
- Git.
- Conexión a Internet para la instalación inicial y las consultas externas.

Clonar el repositorio:

```bash
git clone https://github.com/ros1ndoo/Trabajo-de-Fin-de-Master.git
cd Trabajo-de-Fin-de-Master
```

En Windows, preparar el entorno desde PowerShell:

```powershell
.\scripts\setup.ps1
```

Iniciar la aplicación:

```powershell
.\scripts\start.ps1
```

La aplicación estará disponible en:

http://127.0.0.1:8501

El repositorio incluye el modelo entrenado y los datasets Gold necesarios para cargar el predictor publicado. No es necesario volver a entrenarlo para utilizar la aplicación.

### Pruebas automatizadas

```bash
python -m pytest -q
python -m ruff check src tests app.py
```

GitHub Actions ejecuta estas comprobaciones utilizando Python 3.10 y 3.12.

## 6. Limitaciones y trabajo futuro

AutoReliability es un prototipo académico funcional con las siguientes limitaciones:

- **Cobertura temporal:** el predictor publicado utiliza especificaciones técnicas hasta 2017 y no está validado para lanzamientos actuales.
- **Representatividad:** existen vehículos e identidades no resueltas que pueden introducir sesgos de selección.
- **Capacidad predictiva:** el baseline utiliza una referencia histórica por marca y categoría, sin distinguir necesariamente entre modelos concretos del mismo grupo.
- **Interpretación:** los recalls no permiten medir directamente la fiabilidad mecánica ni determinar el estado de una unidad individual.
- **Validación:** los errores varían entre grupos y no se ha demostrado una mejora predictiva homogénea respecto al baseline.

Las futuras líneas de desarrollo incluyen ampliar la cobertura técnica, mejorar la representación de vehículos sin campañas registradas, resolver los cruces de identidades pendientes y evaluar nuevos modelos mediante validación temporal independiente.

## 7. Conclusión

AutoReliability integra adquisición y procesamiento de datos, ingeniería de características, entrenamiento de modelos y desarrollo de una aplicación interactiva.

El resultado es una herramienta que permite consultar y contextualizar información histórica sobre recalls de seguridad, proporcionando una referencia estadística complementaria para investigar vehículos antes de su compra.

El proyecto demuestra la aplicación práctica de técnicas de Data Science a un problema real, manteniendo explícitas las limitaciones de los datos y del modelo predictivo.

---

**AutoReliability — Trabajo de Fin de Máster en Data Science.**
