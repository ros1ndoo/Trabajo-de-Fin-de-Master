## 1. Idea seleccionada
**Predicción de características y fiabilidad en nuevos lanzamientos automovilísticos**

* **Problema que resuelve:** En la industria automotriz actual, marcada por constantes cambios tecnológicos, los consumidores y las empresas a menudo se enfrentan a decisiones de compra guiadas por el marketing en lugar de por datos objetivos. El problema principal es la asimetría de información al realizar grandes inversiones, como la compra de un vehículo particular o la renovación de una flota comercial. Resolver esto aportaría un gran impacto económico y protegería al consumidor al proporcionar transparencia sobre lo que realmente está adquiriendo.

* **Solución planteada:** Planteo resolver este problema desde un enfoque de Data Science mediante la creación de un modelo predictivo que analice datos históricos. El enfoque general consistirá en utilizar el historial de las marcas, las especificaciones técnicas de motores anteriores y los registros oficiales de *recalls* para estimar anticipadamente la **propensión a fallos de seguridad** de los nuevos modelos antes de que acumulen años en el mercado. Es importante precisar que los *recalls* actúan como un **proxy** legal y observable de la fiabilidad, no como una medida directa de desgaste mecánico cotidiano; esta distinción se mantendrá explícita en todas las capas del proyecto.

* **MVP del proyecto final:** El producto mínimo viable que presentaré al final del curso consistirá en un *dashboard* interactivo o aplicación web. En él, podrá verse funcionando un recomendador/predictor donde el usuario seleccionará un lanzamiento automovilístico reciente del mercado estadounidense y el sistema devolverá una estimación del índice de propensión a recalls, acompañada de una visualización interactiva que compare sus características técnicas con la media de su categoría.

## 2. Datos necesarios
Para desarrollar esta idea, he identificado las siguientes necesidades de datos:
* Necesitaré variables técnicas (cilindrada, tipo de combustible, potencia, peso, transmisión) y variables de fiabilidad (número de *recalls* oficiales ponderados por gravedad). Los *recalls* se usan como proxy observable, no como medida directa de fiabilidad mecánica general.
* El nivel de granularidad adecuado será el detalle por marca, modelo específico y año de fabricación, acotado al **mercado estadounidense** para facilitar el cruce de fuentes.
* La profundidad histórica necesaria deberá abarcar desde 1995 hasta la actualidad para poder analizar el ciclo de vida completo y la evolución de las familias de motores.
* Un volumen aproximado razonable para que el proyecto tenga sentido rondaría entre decenas de miles y cientos de miles de registros, con múltiples modelos por cada año y marca.
* Los datos imprescindibles son las especificaciones técnicas básicas de los vehículos y los registros oficiales de *recalls* de la NHTSA.

## 3. Fuentes de datos previstas
* Las fuentes concretas previstas incluyen:
  * **API de la NHTSA (National Highway Traffic Safety Administration):** Para datos de *recalls* oficiales en EE. UU. Es una fuente gubernamental estable con cobertura garantizada desde 1995.
  * **Dataset técnico de Kaggle "Car Features and MSRP" (publicado por CooperUnion):** Base de datos estática con especificaciones técnicas concretas (potencia, cilindros, categoría) filtradas para el mercado norteamericano. Se trata de una fuente identificada y estable, no una referencia genérica a Kaggle.
* Se trata de fuentes abiertas, públicas y accesibles sin restricciones relevantes.
* El formato esperado de los datos es JSON (para las consultas a la API) y CSV para la base de datos técnica estática.
* Existe un amplio histórico disponible y documentado en estas plataformas institucionales.
* La fuente principal (NHTSA) es altamente estable y mantenida al ser una agencia gubernamental estadounidense.
* **Riesgos detectados:**
  * La falta de estandarización en la nomenclatura entre la base técnica y los registros de la NHTSA (diferencias en cómo se escribe un mismo modelo según la fuente) es el **mayor riesgo técnico**. Se mitigará con un algoritmo de *Fuzzy Matching* auditado con umbrales explícitos.
  * Si el cruce entre fuentes resulta inviable (pérdida superior al 40% de registros), se activará un plan de contingencia basado exclusivamente en el dataset de Kaggle para predecir la depreciación económica del vehículo.
  * Para acotar el riesgo de nomenclaturas dispares, el proyecto se restringe estrictamente al mercado estadounidense.

## 4. Consideraciones de privacidad y protección de datos
* Los datos utilizados se centran en especificaciones de máquinas y registros de fallos de fabricación, por lo que no incluyen información personal identificable.
* No será necesario anonimizar, agregar o filtrar información personal de usuarios.
* Los datos pueden usarse de forma totalmente segura para un proyecto académico, sin infringir normativas de privacidad.
* No existen riesgos éticos o legales que deban tenerse en cuenta, siempre y cuando las predicciones se presenten como **estimaciones estadísticas basadas en proxies** y no como una afirmación rotunda de fiabilidad mecánica garantizada.
* He decidido evitar hacer *web scraping* en foros privados de quejas de usuarios particulares precisamente para no tratar datos personales y garantizar la mayor objetividad posible.

## 5. Viabilidad inicial del proyecto
* Parece totalmente viable obtener los datos necesarios, ya que las fuentes gubernamentales y bases de datos abiertas del sector automotriz son abundantes.
* La información disponible cuenta con suficiente calidad, la granularidad exacta requerida (modelo/año) y una profundidad histórica de varias décadas.
* La idea puede desarrollarse de forma realista durante el curso, acotando el alcance al mercado estadounidense.
* La parte del proyecto que veo más arriesgada en este momento es la limpieza de datos y el cruce entre la base de datos técnica y la base de datos de *recalls*, debido a las posibles inconsistencias en los nombres comerciales de los coches. El *Fuzzy Matching* debe quedar auditado con criterios de aceptación explícitos.
* La alternativa, si el cruce de fuentes no funciona como espero, sería utilizar exclusivamente el dataset técnico de Kaggle para predecir la depreciación económica del vehículo, adaptando ligeramente el MVP.