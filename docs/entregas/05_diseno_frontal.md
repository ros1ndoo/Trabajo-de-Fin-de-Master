# Entrega 5. Diseño del frontal y funcionamiento final de AutoReliability

## 1. Resumen de la solución y del usuario

### 1.1. Problema y utilidad de la aplicación

AutoReliability integra especificaciones técnicas de vehículos y campañas oficiales de seguridad procedentes de fuentes con estructuras y denominaciones diferentes. Su finalidad es facilitar la consulta de esa información, contextualizarla y conservar resultados con una procedencia identificable.

Los usuarios previstos son particulares y gestores de flotas interesados en investigar un modelo. La aplicación permite consultar registros oficiales, obtener una referencia estadística histórica, comparar resultados y descargar un informe.

Estas capacidades están implementadas. Sin embargo, todavía no se ha realizado un estudio con usuarios que demuestre ahorro de tiempo, reducción de errores o mejora de las decisiones de compra. Estos beneficios se consideran hipótesis pendientes de validación.

La cobertura técnica de la versión final comprende vehículos del mercado estadounidense entre 1995 y 2017. La disponibilidad de registros oficiales de años posteriores no amplía automáticamente la cobertura del predictor.

### 1.2. Resultado principal y significado del índice

La aplicación devuelve `prediccion_indice_100`, una estimación de un índice construido a partir de campañas de seguridad de NHTSA.

El objetivo se obtiene mediante:

1. La identificación y deduplicación de campañas.
2. La suma de sus pesos dentro de una ventana común de tres años calendario desde el año-modelo.
3. La normalización respecto a estadísticas de categoría calculadas con datos de entrenamiento.
4. La transformación y acotación del resultado a una escala de 0 a 100.

La transformación es:

`índice = clip(50 − 10 × ((carga ponderada − media de entrenamiento de la categoría) / desviación de entrenamiento de la categoría), 0, 100)`

Los pesos de las campañas proceden de reglas textuales del proyecto. Constituyen una aproximación metodológica y no una clasificación oficial de gravedad.

Una puntuación mayor indica una referencia más favorable en el índice de campañas de seguridad. No representa:

- Un porcentaje de fiabilidad.
- Una probabilidad individual de avería.
- Una garantía de seguridad.
- Una evaluación del mantenimiento o del estado real de una unidad.
- Una recomendación automática de compra.

Como el objetivo se normaliza por categoría, una puntuación superior entre vehículos de categorías distintas no implica necesariamente un menor número absoluto de campañas.

### 1.3. Modelo operativo de la versión final

El artefacto seleccionado utiliza un **baseline histórico de medias por marca y categoría**, implementado en `BrandSegmentMeanBaseline`.

Ridge y Random Forest se evaluaron como candidatos, pero no son el modelo operativo de la publicación final.

El baseline utiliza la media aprendida del índice para la combinación de `marca` y `categoria_vehiculo`. Cuando falta esa combinación, puede recurrir a la referencia de marca. La ruta web exige que la marca esté representada en el baseline entrenado y rechaza las marcas sin soporte, en lugar de servirles silenciosamente una media global.

El contrato común de entrada del pipeline contiene:

- `marca`
- `categoria_vehiculo`
- `mediana_cilindros`
- `mediana_cv`
- `hist_fiabilidad_marca`

La presencia de una variable en este contrato no implica que el modelo seleccionado la utilice. En el baseline publicado, **potencia, cilindros e historial reciente no modifican la puntuación**.

Modelo y año identifican la ficha consultada y permiten contextualizar el resultado, pero no individualizan el cálculo dentro de un mismo grupo entrenado. Por ello, distintos modelos o años pueden recibir la misma estimación.

### 1.4. Selección del modelo y resultados

La fuente de referencia de las métricas es [artifacts/model_metrics.json](../../artifacts/model_metrics.json). Las cifras siguientes corresponden al artefacto congelado el 25 de septiembre de 2026 y se transcriben de ese archivo.

| Candidato | MAE de validación | RMSE de validación | Casos |
| --- | ---: | ---: | ---: |
| Baseline histórico | 10,5771 | 13,7543 | 90 |
| Random Forest | 10,8236 | 13,5027 | 90 |
| Ridge | 10,9386 | 13,4978 | 90 |

El baseline obtuvo el menor **MAE de validación**, criterio principal de selección. No obtuvo el menor RMSE: Ridge y Random Forest registraron valores algo inferiores en esa métrica. Por tanto, no se afirma que el baseline sea superior en todas las medidas.

Random Forest redujo el MAE de Ridge un 1,0513 %, por debajo del requisito de mejora estrictamente superior al 10 % establecido en el protocolo. Tampoco superó al baseline en MAE.

La evaluación final del artefacto publicado registra:

| Métrica de test | Resultado |
| --- | ---: |
| Casos evaluados | 819 |
| MAE | 12,1537 |
| RMSE | 16,9991 |

Las particiones efectivas utilizadas en la evaluación son:

| Partición | Años | Casos |
| --- | --- | ---: |
| Entrenamiento para selección | 1995–2005 | 287 |
| Validación | 2008–2009 | 90 |
| Test | 2012–2017 | 819 |

Las restricciones de madurez de las etiquetas explican los intervalos entre particiones. Tras seleccionar el candidato, el ajuste final combina entrenamiento y validación, con 377 registros, sin incorporar el test.

El test ya ha sido examinado. Estas métricas describen el rendimiento sobre la población histórica evaluada y no garantizan el mismo error para todas las marcas, categorías o vehículos.

El JSON de métricas y la versión del artefacto constituyen la referencia para cualquier cifra reproducida en otros documentos. Esta tabla es una transcripción de la versión indicada, no una sincronización automática.

## 2. Diseño del frontal y evolución del mockup

El [mockup original](../assets/05_mockup_frontal.png) se conserva como antecedente conceptual. No debe utilizarse como captura de la aplicación final ni como evidencia de sus resultados.

Sus ejemplos de vehículos recientes, cifras ilustrativas, contribuciones de Ridge y representación del MAE como intervalo corresponden al planteamiento inicial y quedan sustituidos por la descripción de esta entrega.

La implementación final utiliza **Streamlit**, gráficos interactivos de **Plotly** y estilos CSS. Bootstrap fue una referencia conceptual del diseño, no el framework de ejecución de la aplicación.

El frontal organiza la información en los siguientes elementos:

| Elemento | Función actual |
| --- | --- |
| Selectores de marca, modelo y año | Identificar la ficha consultada mediante opciones en cascada |
| Indicador de 0 a 100 | Mostrar la estimación del baseline y el método utilizado |
| MAE temporal | Informar del error absoluto medio de evaluación del artefacto |
| Factores explicativos | Identificar la referencia de grupo y su diferencia respecto a la media global de entrenamiento |
| Perfil técnico frente al segmento | Aportar contexto descriptivo sobre las características disponibles |
| Contexto histórico | Mostrar información de cohortes anteriores |
| Consulta oficial NHTSA | Consultar campañas con indicación de fuente y fecha |
| Comparación | Contrastar descriptivamente hasta dos resultados |
| Exportación CSV/PDF | Conservar el resultado, su procedencia y sus limitaciones |

Los paneles de inventario complementario EPA/NHTSA y revisión de exclusiones no forman parte del frontal visible final. Sus datos y controles internos se conservan.

También se han retirado las etiquetas textuales de tramos del índice. Los colores del indicador no constituyen umbrales de seguridad validados.

## 3. Justificación del diseño

### 3.1. Utilidad y valor de la solución

La aportación funcional del frontal consiste en reunir la consulta técnica, la evidencia oficial y una referencia estadística en un recorrido accesible.

La aplicación permite distinguir entre información observada, información insuficiente y fallos de consulta. También facilita guardar resultados con referencias de procedencia.

El índice es un elemento complementario. La consulta oficial y las exportaciones mantienen utilidad aunque el usuario no utilice la puntuación.

No se presenta el resultado como una recomendación de inversión ni se afirma que permita evitar gastos de mantenimiento o averías. La utilidad real para los usuarios deberá comprobarse mediante una evaluación específica.

### 3.2. Flujo de usuario

1. **Selección del vehículo.** El usuario elige marca, modelo y año. Las opciones se actualizan en cascada.

2. **Consulta del catálogo.** El servicio localiza la ficha en el catálogo, que integra datos de Gold y del catálogo de inferencia. Aparecer en los selectores no garantiza disponer de una etiqueta propia de recalls ni de soporte suficiente para obtener una estimación.

3. **Comprobación de soporte.** La ruta web verifica que la marca esté representada en el baseline entrenado. Si falta soporte, comunica la limitación y mantiene disponible la consulta oficial.

4. **Consulta de evidencia oficial.** El servicio intenta recuperar campañas NHTSA e identifica su procedencia. Si no existe una correspondencia suficientemente verificada entre denominaciones, informa de esa limitación.

5. **Cálculo del índice.** Se aplica el artefacto publicado. No se reentrena durante la solicitud y los recalls propios recuperados en ese momento no se incorporan como características predictoras ni como nuevas etiquetas del modelo congelado.

6. **Presentación del resultado.** Se muestran el índice, el método, el MAE cuando corresponde, la explicación del grupo utilizado y el contexto técnico e histórico.

7. **Acciones posteriores.** El usuario puede comparar resultados, descargar un resumen, cambiar la selección o reiniciar el formulario. Los cambios en los selectores invalidan el resultado anterior para evitar atribuirlo a otro vehículo.

### 3.3. Estados y tratamiento de errores

La aplicación diferencia estas situaciones:

- Campañas observadas para una identidad y un periodo determinados.
- Ausencia de campañas dentro de una ventana válida.
- Consulta no disponible o fallida.
- Identidad o correspondencia insuficientemente verificada.
- Falta de soporte para una estimación.

Un fallo de API no equivale a cero campañas. Una respuesta vacía tampoco acredita por sí sola ausencia de recalls.

Si existe una instantánea oficial aplicable como reserva, el sistema puede utilizarla indicando su fuente y fecha. Si no dispone de evidencia recuperable, informa de la indisponibilidad.

Una marca con soporte en el baseline puede conservar su estimación local aunque falle la consulta oficial. Ambas salidas permanecen diferenciadas.

Existe además una referencia histórica de contingencia para determinadas situaciones sin artefacto válido. Esta salida se identifica como `fallback` y no debe confundirse con el baseline entrenado que ganó la selección. No hereda su MAE y no genera una puntuación cuando faltan cohortes anteriores suficientes.

### 3.4. Experiencia de usuario

- **Claridad:** el resultado se acompaña del método utilizado y de advertencias sobre su interpretación.
- **Separación conceptual:** la explicación del modelo y el perfil técnico cumplen funciones diferentes.
- **Control:** el usuario puede modificar la selección, reiniciar la consulta y guardar comparaciones.
- **Coherencia:** la identidad mostrada en el resultado debe coincidir con la seleccionada.
- **Adaptación:** la distribución prioriza el escritorio y adapta las columnas a pantallas estrechas.
- **Prudencia:** los colores y los decimales no se interpretan como evidencia de seguridad ni como precisión individual.

El diseño responsivo no equivale a una certificación formal de accesibilidad ni a una validación de usabilidad con usuarios.

## 4. Presentación de resultados y explicabilidad

### 4.1. Explicación del modelo seleccionado

La explicación del baseline es determinista. Identifica el grupo utilizado, muestra su estimación y calcula la diferencia respecto a la media global aprendida.

Esa diferencia es descriptiva. No representa un efecto causal ni identifica mecanismos de avería.

La salida del modelo activo no utiliza coeficientes Ridge, importancias de Random Forest o valores SHAP para justificar la puntuación. El código puede conservar métodos de atribución para los candidatos evaluados, pero estos no constituyen la explicación del baseline publicado.

### 4.2. Papel del radar técnico

El radar compara características disponibles del vehículo con una referencia de su segmento. Sus ejes se escalan para facilitar la visualización y los valores técnicos pueden consultarse de forma interactiva.

Su función es **contextual**, no explicativa de la predicción.

Potencia, cilindros e historial reciente no han producido la estimación del baseline actual. Por tanto:

- Una mayor potencia no implica una contribución positiva o negativa al índice.
- Una mayor superficie en el radar no significa mayor fiabilidad o mayor riesgo.
- Las diferencias visuales no acreditan significación estadística.
- Las características mostradas no permiten identificar causas de averías.

La lectura del resultado del baseline explicita que estas variables no intervienen en su cálculo. El radar y el bloque de factores deben interpretarse como componentes distintos.

### 4.3. Contexto histórico de marca

La variable `hist_fiabilidad_marca` utiliza cohortes de lanzamiento anteriores cuya ventana de observación ya se ha completado.

Con la configuración actual, considera cohortes con lanzamiento entre año−5 y año−3. No equivale al número de campañas notificadas durante los tres años calendario inmediatamente anteriores a la consulta.

Cuando se utiliza una referencia de mercado por falta de historial suficiente de marca, se identifica su procedencia. Este contexto no modifica la puntuación del baseline seleccionado.

### 4.4. Interpretación del MAE

El MAE es el promedio de las diferencias absolutas entre estimaciones y etiquetas observadas en la población evaluada.

El valor mostrado corresponde a la evaluación del artefacto, no al error particular del vehículo seleccionado. Por ello:

- No se presenta como `puntuación ± MAE`.
- No constituye un intervalo de confianza.
- No constituye un intervalo predictivo individual.
- No establece un límite máximo de error.
- No permite afirmar que pequeñas diferencias entre puntuaciones sean significativas.

Si una salida de contingencia no dispone de evaluación propia, el MAE se muestra como no disponible.

La heterogeneidad entre grupos, las exclusiones y la cobertura de las fuentes limitan la generalización del error global.

### 4.5. Narración y uso de IA

La explicación del baseline se genera mediante reglas y valores verificados. No necesita un modelo de lenguaje ni utiliza IA generativa para proponer causas mecánicas.

El proyecto conserva un componente opcional de selección de cláusulas mediante un modelo local para otros tipos de atribuciones. La rama del baseline lo evita expresamente.

Por tanto, la IA generativa no se presenta como una función operativa de explicación del artefacto final.

### 4.6. Temporalidad y trazabilidad

El pipeline aplica separación temporal, restricciones de madurez de etiquetas y transformaciones ajustadas con entrenamiento. Las variables de recalls propias que construyen el objetivo se excluyen de las características predictoras.

Estos controles reducen riesgos concretos de fuga de información. No demuestran que todas las especificaciones y correcciones de las fuentes estuvieran disponibles exactamente en cada fecha histórica de lanzamiento.

La evaluación se presenta como retrospectiva. Las consultas anteriores a la disponibilidad del modelo se identifican como tales y no se describen como predicciones realizadas en el momento del lanzamiento.

Los informes incorporan identidad del vehículo, método, fecha, referencias de versión y evidencia oficial disponible. La reproducción completa requiere conservar también los datos y artefactos correspondientes.

## 5. Alcance final y mejoras futuras

### 5.1. Funciones implementadas

- Interfaz web con selectores en cascada.
- Inferencia con el baseline seleccionado.
- Controles de soporte y tratamiento explícito de información insuficiente.
- Consulta oficial bajo demanda y alternativas con evidencia almacenada identificable.
- Explicación determinista de la referencia de grupo.
- Contexto técnico e histórico diferenciado de los factores del modelo.
- Comparación descriptiva de resultados.
- Exportaciones CSV y PDF.
- Publicación versionada de artefactos y comprobaciones de integridad.
- Pruebas automatizadas del procesamiento, el servicio y la interfaz.

### 5.2. Limitaciones y objetivos no demostrados

Los candidatos más complejos no demostraron una mejora suficiente frente al baseline bajo el criterio principal.

Tampoco se ha demostrado:

- Una mejora de las decisiones de usuarios reales.
- Un rendimiento homogéneo entre marcas y categorías.
- Una probabilidad individual de avería calibrada.
- Capacidad para recomendar la compra de un vehículo concreto.
- Generalización a lanzamientos actuales.
- Una explicación causal de fallos mecánicos.

La consulta se realiza por modelo y año, no por VIN. El scraping de precios de segunda mano y el cálculo de depreciación quedan fuera del alcance de la aplicación.

### 5.3. Prioridades de evolución

1. **Validar utilidad con usuarios:** medir comprensión del índice, tiempo de consulta y errores de interpretación frente al procedimiento habitual.

2. **Mejorar la evidencia disponible:** resolver correspondencias e investigar exclusiones sin convertir datos desconocidos en ceros ni rebajar los requisitos de calidad.

3. **Ampliar cobertura técnica:** incorporar fuentes recientes y verificar su compatibilidad e identidad antes de utilizarlas.

4. **Evaluar modelos más individualizados:** exigir mejoras comprobables mediante un protocolo previo y datos adicionales no utilizados para orientar los cambios.

5. **Estudiar incertidumbre predictiva:** desarrollar y validar métodos específicos, sin utilizar el MAE como sustituto de un intervalo individual.

6. **Mantener coherencia documental:** generar los extractos numéricos de la documentación a partir de las métricas del artefacto correspondiente. Esta automatización evitaría divergencias entre documentos, pero no se presupone implementada en la versión actual.

## 6. Fuentes de verificación

- [Métricas del modelo](../../artifacts/model_metrics.json): selección, resultados de candidatos, test, particiones y normalización.
- [Cierre de entrega](../../artifacts/delivery_freeze.json): identificación y huellas de los artefactos.
- [Publicación activa](../../releases/active.json): versión utilizada por el servicio.
- [Modelado](../../src/auto_reliability/modeling.py): baseline y protocolo de selección.
- [Servicio](../../src/auto_reliability/service.py): soporte de inferencia, consulta oficial y contingencias.
- [Explicabilidad](../../src/auto_reliability/explainability.py): explicación del grupo y comparación con la media global.
- [Narración](../../src/auto_reliability/narrative.py): salida determinista del baseline.
- [Frontal](../../src/auto_reliability/dashboard.py): presentación, interpretación, radar y comparación.
- [Auditorías temporales congeladas](../../artifacts/frozen_evidence/temporal_results.json): análisis descriptivos y resultados por subgrupos.

Esta entrega documenta el comportamiento del artefacto final. Su actualización no modifica el modelo, las métricas, los datos ni los resultados de evaluación.
