# Guia de entrevista: Electricity Load Forecasting

Este documento es para que puedas explicar el proyecto con seguridad, sin memorizar codigo linea por linea. La idea principal es: **construiste un pipeline reproducible para predecir demanda electrica horaria usando datos historicos, clima y calendario, comparando modelos simples contra modelos mas avanzados sin contaminar el test con informacion del futuro.**

## 1. Resumen en 30 segundos

> Este proyecto predice la demanda electrica nacional (`nat_demand`) una hora hacia adelante. Uso datos horarios historicos de demanda, variables meteorologicas y senales de calendario. El flujo esta pensado como un proyecto real de machine learning: ordeno los datos por tiempo, hago split cronologico en train, validation y test, ajusto los scalers solo con train para evitar data leakage, creo ventanas de 168 horas y comparo baselines interpretables contra un LSTM. Tambien dejo metricas y un dashboard en Streamlit para comunicar resultados.

## 2. Que problema resuelve

La demanda electrica cambia por hora, dia de semana, clima, feriados y patrones operativos. Poder anticiparla ayuda a planificar generacion, compra de energia, estabilidad de red y capacidad operativa.

En este repo el problema esta planteado como **regresion de series temporales one-step-ahead**:

- **Entrada:** las ultimas 168 horas de informacion.
- **Salida:** demanda electrica de la siguiente hora.
- **Target:** `nat_demand`.
- **Frecuencia:** horaria.
- **Datos:** `data/raw/continuous_dataset.csv`.

## 3. Mapa mental del proyecto

```text
data/raw/continuous_dataset.csv
        |
        v
src/data.py
load_dataset -> add_time_features -> split_chronologically -> scalers -> create_sequences
        |
        v
src/train.py                         src/evaluate.py
entrena LSTM con validation           compara naive, tabular models y LSTM si existe checkpoint
        |                              |
        v                              v
models/lstm_forecast.pt              reports/metrics.csv
                                      reports/figures/lstm_test_forecast.png
        |
        v
app/streamlit_app.py
dashboard visual para explicar metricas y forecast
```

## 4. Componentes importantes

### `src/features.py`

Aqui se define el target, columnas de clima, columnas de calendario y funciones de feature engineering.

Puntos que debes saber explicar:

- `add_time_features` crea senales de hora, dia de semana y mes.
- Usa `sin` y `cos` para variables ciclicas, porque la hora 23 y la hora 0 estan cerca.
- `create_sequences` transforma la serie en ventanas: 168 horas pasadas para predecir la hora siguiente.

### `src/data.py`

Aqui esta la preparacion reproducible de datos.

Puntos clave:

- `load_dataset` lee el CSV, convierte `datetime` y ordena temporalmente.
- `split_chronologically` separa train, validation y test sin mezclar el tiempo.
- `prepare_data` ajusta `MinMaxScaler` solo en train y luego transforma validation/test.
- Esto evita leakage: el modelo no ve estadisticas del futuro.

### `src/baselines.py`

Aqui estan los modelos de comparacion.

Baselines implementados:

- `naive_previous_day`: usa la demanda de la misma hora del dia anterior.
- `seasonal_naive_previous_week`: usa la misma hora de la semana anterior.
- `linear_regression`: modelo interpretable con lags y medias moviles.
- `random_forest`: baseline no lineal fuerte.
- `xgboost`: opcional si esta instalado.

Frase buena para entrevista:

> No quise saltar directo al LSTM. Primero construi baselines simples y fuertes para saber si el modelo avanzado realmente agrega valor.

### `src/models.py`

Define `LSTMForecast`.

Idea principal:

- La LSTM recibe una secuencia de horas.
- Procesa dependencias temporales.
- Toma el ultimo estado temporal y lo pasa por una capa lineal.
- Devuelve una prediccion continua de demanda.

### `src/train.py`

Entrena el LSTM usando train y validation.

Puntos clave:

- Usa `MSELoss`.
- Usa `Adam`.
- Guarda el mejor modelo segun `validation_loss`.
- Guarda metadata en JSON para reproducibilidad.

### `src/evaluate.py`

Evalua modelos en el test final.

Puntos clave:

- Calcula MAE, RMSE y MAPE.
- Evalua baselines siempre.
- Evalua LSTM solo si existe `models/lstm_forecast.pt`.
- Guarda resultados en `reports/metrics.csv`.

## 5. Resultados actuales

El archivo `reports/metrics.csv` actualmente contiene resultados de baselines:

| Modelo | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| random_forest | 23.96 | 34.54 | 2.02% |
| linear_regression | 29.68 | 38.77 | 2.51% |
| seasonal_naive_previous_week | 71.14 | 99.16 | 5.94% |
| naive_previous_day | 73.33 | 110.60 | 5.99% |

Interpretacion:

- `random_forest` es el mejor baseline actual.
- Los baselines naive sirven como piso minimo razonable.
- La mejora de random forest contra naive muestra que las features de clima, calendario, lags y rolling means aportan informacion.
- En el estado actual del repo no hay checkpoint de LSTM en `models/`, por eso el CSV no incluye una fila `lstm`.

Respuesta honesta si te preguntan:

> El pipeline del LSTM esta implementado, pero el reporte actual que tengo versionado muestra baselines. Para incluir el LSTM tengo que ejecutar `python -m src.train --epochs 30` y luego `python -m src.evaluate`.

## 6. Como explicarlo paso a paso en entrevista

### Paso 1: Contexto

> Queria resolver un problema de forecasting aplicado a energia: predecir demanda electrica horaria usando historia, clima y calendario.

### Paso 2: Datos

> El dataset tiene observaciones horarias desde 2015 hasta 2020. La variable objetivo es `nat_demand`. Tambien uso variables meteorologicas de tres ubicaciones y flags como feriados y escuela.

### Paso 3: Preparacion

> Como es una serie temporal, no uso random split. Ordeno los datos por fecha y separo train, validation y test cronologicamente. Ademas, ajusto los scalers solo en train para no filtrar informacion del futuro.

### Paso 4: Features

> Agrego features ciclicas de hora, dia de semana y mes con seno/coseno. Tambien uso demanda historica y variables de clima. Para los modelos tabulares creo lags de 1, 24 y 168 horas y medias moviles de 24 y 168 horas.

### Paso 5: Modelos

> Compare modelos simples, tabulares y una arquitectura LSTM. Los modelos naive me dan un piso. Linear regression da interpretabilidad. Random forest captura no linealidades. LSTM esta pensado para aprender patrones secuenciales usando las ultimas 168 horas.

### Paso 6: Evaluacion

> Uso MAE, RMSE y MAPE. MAE es facil de interpretar en unidades de demanda. RMSE penaliza errores grandes. MAPE ayuda a comunicar el error porcentual.

### Paso 7: Cierre

> Lo mas importante no es solo el modelo, sino la disciplina del pipeline: split temporal, evitar leakage, comparar contra baselines y reportar resultados de forma reproducible.

## 7. Preguntas y respuestas para practicar

### 1. Por que elegiste este proyecto?

Porque combina machine learning con un caso real de energia. La demanda electrica es importante para planificacion operativa, costos y estabilidad de red. Tambien es un buen problema para demostrar series temporales, feature engineering, validacion temporal y comunicacion de resultados.

### 2. Cual es la variable objetivo?

La variable objetivo es `nat_demand`, que representa la demanda electrica nacional por hora.

### 3. Que significa one-step-ahead forecasting?

Significa que el modelo predice solo el siguiente punto temporal. En este caso, usa las ultimas 168 horas para predecir la demanda de la proxima hora.

### 4. Por que usaste 168 horas?

168 horas son 7 dias. En demanda electrica suelen existir patrones diarios y semanales: la misma hora de dias anteriores y de la semana anterior puede ser muy informativa.

### 5. Por que no hiciste random train/test split?

Porque en series temporales un split aleatorio mezcla pasado y futuro. Eso puede hacer que el modelo aprenda indirectamente informacion futura y que las metricas parezcan mejores de lo que serian en produccion.

### 6. Que es data leakage en este proyecto?

Seria usar informacion del futuro durante entrenamiento o preparacion. Por ejemplo, ajustar el scaler con todo el dataset antes de dividirlo. En este proyecto evito eso ajustando los scalers solo con train.

### 7. Por que hiciste baselines?

Porque un modelo avanzado solo vale la pena si supera alternativas simples. En forecasting, un baseline naive como "misma hora de ayer" o "misma hora de la semana pasada" puede ser sorprendentemente fuerte.

### 8. Que modelo funciona mejor actualmente?

En las metricas actuales, `random_forest` tiene el mejor MAE, RMSE y MAPE entre los baselines reportados. Tiene MAE aproximado de 23.96 y MAPE de 2.02%.

### 9. Por que random forest puede funcionar bien?

Porque puede capturar relaciones no lineales entre clima, calendario y demanda, y tambien usar lags y medias moviles sin requerir una arquitectura secuencial profunda.

### 10. Que hace una LSTM?

Una LSTM es una red recurrente disenada para secuencias. Mantiene una memoria interna que ayuda a aprender dependencias temporales. En este proyecto toma una ventana de 168 horas y predice la demanda de la siguiente hora.

### 11. Por que escalar los datos?

Porque la LSTM entrena mejor cuando las variables estan en rangos similares. Se usa `MinMaxScaler` para features y target. Importante: se ajusta solo con train.

### 12. Que diferencia hay entre validation y test?

Validation se usa para elegir el mejor modelo durante entrenamiento. Test se deja intacto hasta el final para estimar como funcionaria el modelo en datos no vistos.

### 13. Que metricas usaste?

MAE, RMSE y MAPE. MAE mide error promedio absoluto, RMSE castiga mas los errores grandes y MAPE expresa error porcentual.

### 14. Que limitaciones tiene el proyecto?

El LSTM todavia debe entrenarse y compararse formalmente en el reporte actual. Tambien seria util agregar forecasting directo de 24 horas, analisis de residuos por hora/dia y validacion walk-forward.

### 15. Que mejorarias?

Agregaria forecasting multi-horizon de 24 horas, busqueda de hiperparametros, validacion walk-forward, analisis de errores por hora y feriados, y adaptacion a datos de Dinamarca/Energinet si la entrevista esta enfocada en energia nordica.

### 16. Como lo llevarias a produccion?

Separaria el pipeline en jobs: ingestion de datos, validacion, feature engineering, entrenamiento periodico, prediccion diaria, monitoreo de error y dashboard. Tambien agregaria tracking de experimentos y alertas si el error aumenta.

### 17. Que aprendiste?

Aprendi que en series temporales la evaluacion importa tanto como el modelo. Si el split o el escalado estan mal, las metricas pueden ser irreales. Tambien aprendi a comparar modelos profundos contra baselines antes de asumir que son mejores.

### 18. Como defenderias que sabes lo que hiciste?

Puedes decir:

> Se como fluye cada parte: datos crudos, features temporales, split cronologico, escalado sin leakage, ventanas de 168 horas, entrenamiento con validation, evaluacion final con MAE/RMSE/MAPE y dashboard para comunicar resultados. Si tuviera mas tiempo, lo extenderia a forecast de 24 horas y validacion walk-forward.

## 8. Frases cortas para sonar claro

- "En series temporales, el orden importa."
- "El test representa futuro no visto."
- "Primero comparo contra baselines para evitar sobreingenieria."
- "El scaler se ajusta solo con train para evitar leakage."
- "168 horas capturan patrones diarios y semanales."
- "MAE se entiende facil; RMSE castiga errores grandes; MAPE comunica porcentaje."
- "El valor del proyecto esta en el pipeline completo, no solo en el modelo."

## 9. Comandos utiles

```bash
pip install -r requirements.txt
python -m src.train --epochs 30
python -m src.evaluate
streamlit run app/streamlit_app.py
```

## 10. Guion final de 2 minutos

> Este proyecto predice demanda electrica horaria. Trabajo con un dataset horario desde 2015 hasta 2020, donde el target es `nat_demand`. Uso demanda historica, clima y calendario. Como es una serie temporal, hice split cronologico en train, validation y test; no random split. Tambien ajuste los scalers solo en train para evitar leakage.
>
> Despues cree features temporales como hora, dia de semana y mes usando seno/coseno, porque son variables ciclicas. Para los modelos tabulares agregue lags y rolling means. Compare baselines naive, linear regression y random forest. Actualmente el mejor baseline reportado es random forest, con MAPE cercano a 2%.
>
> Tambien implemente una LSTM que usa 168 horas de historia para predecir la siguiente hora. La razon de 168 es que representa una semana completa y captura patrones diarios y semanales. El entrenamiento guarda el mejor checkpoint segun validation loss y la evaluacion final se hace sobre test.
>
> Para mi, lo mas importante del proyecto es que muestra buenas practicas de forecasting: evitar leakage, evaluar cronologicamente, comparar contra baselines y comunicar resultados con metricas y dashboard. Como siguiente paso lo extenderia a prediccion directa de 24 horas y validacion walk-forward.
