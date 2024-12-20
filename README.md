# KingsLeagueML

# Kings League: Análisis de Datos y Estadísticas Avanzadas

## Introducción
La Kings League es mucho más que un simple torneo de fútbol; es un fenómeno que combina innovación, emoción y estrategia en cada partido. Con elementos únicos como el dado que define las reglas del juego en tiempo real, esta liga nos invita a explorar un terreno inédito en el análisis de datos deportivos. Este proyecto nace con el objetivo de desentrañar los patrones ocultos, generar estadísticas clave y ofrecer una perspectiva única sobre los factores que definen el rendimiento de los equipos.

## Descripción del Proyecto
Este proyecto se centra en analizar los datos del primer split de la Kings League, abarcando desde la jornada 3 hasta la jornada 11. Usando herramientas avanzadas de análisis de datos y visualización, buscamos responder preguntas como:

- ¿Cómo influye el número que sale en el dado en los resultados de los partidos?
- ¿Qué equipos tienen un mejor rendimiento en diferentes modalidades de juego?
- ¿Cómo afecta el tiempo jugado tras la tirada del dado en el marcador final?

## Datos
El dataset utilizado incluye las siguientes columnas clave:

- **match_id**: Identificador único del partido.
- **date**: Fecha del partido en formato día/mes/año.
- **team1** y **team2**: Equipos enfrentados.
- **goals_team1** y **goals_team2**: Goles anotados antes de la tirada del dado.
- **dado_number**: Número obtenido en el dado (define la modalidad del juego, como 5 vs 5).
- **minutes_after_dice_roll**: Minutos jugados en la modalidad definida por el dado.
- **goals_team1_after_dice** y **goals_team2_after_dice**: Goles anotados por cada equipo durante esta modalidad.
- **Total_goals_in_dice**: Goles totales anotados durante el tiempo definido por el dado.

## Objetivos
1. **Estadísticas por equipo**: Crear un análisis individualizado para cada equipo de la liga.
2. **Estrategias basadas en el dado**: Identificar patrones de rendimiento según el número obtenido en el dado.
3. **Visualizaciones interactivas**: Presentar los resultados de forma atractiva y comprensible para facilitar la toma de decisiones.

## Tecnologías y Herramientas
- **Python**: Para limpieza de datos, análisis estadístico y modelado.
- **Pandas y NumPy**: Procesamiento y manipulación de datos.
- **Matplotlib y Seaborn**: Creación de visualizaciones estáticas.
- **Plotly y Dash**: Visualizaciones interactivas.
- **Jupyter Notebook**: Desarrollo iterativo del análisis.

## Resultados Esperados
Este proyecto busca proporcionar un entendimiento profundo y accionable sobre los datos de la Kings League. Las visualizaciones y el análisis permitirán a fanáticos, analistas y estrategas tomar decisiones informadas y disfrutar de una nueva perspectiva sobre este emocionante torneo.

---

