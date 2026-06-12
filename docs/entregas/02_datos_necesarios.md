## 1. Idea seleccionada
**Predicción de características y fiabilidad en nuevos lanzamientos automovilísticos**

* **Problema que resuelve:** En la industria automotriz actual, marcada por constantes cambios tecnológicos, los consumidores y las empresas a menudo se enfrentan a decisiones de compra guiadas por el marketing en lugar de por datos objetivos. El problema principal es la asimetría de información al realizar grandes inversiones, como la compra de un vehículo particular o la renovación de una flota comercial. Resolver esto aportaría un gran impacto económico y protegería al consumidor al proporcionar transparencia sobre lo que realmente está adquiriendo.

* **Solución planteada:** Planteo resolver este problema desde un enfoque de Data Science mediante la creación de un modelo predictivo que analice datos históricos. El enfoque general consistirá en utilizar el historial de las marcas, las especificaciones técnicas de motores anteriores y los índices de fallos para estimar anticipadamente la fiabilidad y las características reales de los nuevos modelos antes de que acumulen años en el mercado.

* **MVP del proyecto final:** El producto mínimo viable que presentaré al final del curso consistirá en un *dashboard* interactivo o aplicación web. En él, podrá verse funcionando un recomendador/predictor donde el usuario seleccionará un lanzamiento automovilístico reciente y el sistema devolverá una estimación de la fiabilidad del motor, acompañada de una visualización interactiva que compare sus características técnicas con la media de su categoría.

## 2. Datos necesarios
Para desarrollar esta idea, he identificado las siguientes necesidades de datos:
* Necesitaré variables técnicas (cilindrada, tipo de combustible, potencia, peso, transmisión) y variables de fiabilidad (número de quejas, gravedad de averías, llamadas a revisión oficiales o *recalls*).
* El nivel de granularidad adecuado será el detalle por marca, modelo específico, motorización y año de fabricación.
* La profundidad histórica necesaria deberá abarcar los últimos 15 a 20 años para poder analizar el ciclo de vida completo y la evolución de las familias de motores.
* Un volumen aproximado razonable para que el proyecto tenga sentido rondaría entre decenas de miles y cientos de miles de registros ,múltiples modelos por cada año y marca.
* Los datos imprescindibles son las especificaciones técnicas básicas de los vehículos y los registros oficiales de *recalls*/fallos. Los datos deseables, pero no obligatorios, serían los costes promedio de mantenimiento en taller por marca.

## 3. Fuentes de datos previstas
* Las fuentes concretas previstas incluyen la API pública de la NHTSA (National Highway Traffic Safety Administration) para datos de fiabilidad y *recalls*, así como repositorios como Kaggle para datasets de especificaciones técnicas de vehículos.
* Se trata de fuentes abiertas, públicas y accesibles sin restricciones relevantes.
* El formato esperado de los datos es JSON (para las consultas a la API) y formato CSV para las bases de datos técnicas estáticas.
* Existe un amplio histórico disponible y documentado en estas plataformas institucionales.
* La fuente principal (NHTSA) parece altamente estable y mantenida al ser una agencia gubernamental.
* Los riesgos detectados incluyen la falta de estandarización en la nomenclatura (ej. diferencias en cómo se nombra un mismo modelo en Europa vs. EE. UU.) y posibles datos incompletos en motorizaciones muy minoritarias.

## 4. Consideraciones de privacidad y protección de datos
* Los datos utilizados se centran en especificaciones de máquinas y registros de fallos de fabricación, por lo que no incluyen información personal identificable.
* No será necesario anonimizar, agregar o filtrar información personal de usuarios.
* Los datos pueden usarse de forma totalmente segura para un proyecto académico, sin infringir normativas de privacidad.
* No existen riesgos éticos o legales que deban tenerse en cuenta, siempre y cuando las predicciones se presenten como estimaciones estadísticas y no como una afirmación rotunda que pueda suponer difamación hacia una marca.
* He decidido evitar hacer *web scraping* en foros privados de quejas de usuarios particulares precisamente para no tratar datos personales y garantizar la mayor objetividad posible.

## 5. Viabilidad inicial del proyecto
* Parece totalmente viable obtener los datos necesarios, ya que las fuentes gubernamentales y bases de datos abiertas del sector automotriz son abundantes.
* La información disponible cuenta con suficiente calidad, la granularidad exacta requerida (modelo/año) y una profundidad histórica de varias décadas.
* La idea puede desarrollarse de forma realista durante el curso, acotando el alcance si fuera necesario.
* La parte del proyecto que veo más arriesgada en este momento es la limpieza de datos y el cruce entre la base de datos técnica y la base de datos de fiabilidad, debido a las posibles inconsistencias en los nombres comerciales de los coches.
* La alternativa, si la fuente principal de datos o la API no funciona como espero, sería utilizar un dataset estático pre-limpiado de Kaggle que contenga tanto especificaciones como precios o valoraciones, adaptando ligeramente el MVP para predecir el "valor" o "depreciación" en lugar del fallo mecánico.