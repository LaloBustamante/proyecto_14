'''
Descripción del proyecto
La compañía Sweet Lift Taxi ha recopilado datos históricos sobre pedidos de taxis en los aeropuertos. Para atraer a más conductores durante 
las horas pico, necesitamos predecir la cantidad de pedidos de taxis para la próxima hora. Construye un modelo para dicha predicción.

La métrica RECM en el conjunto de prueba no debe ser superior a 48.

Instrucciones del proyecto

1. Descarga los datos y remuestréalos de tal forma que cada punto de datos de los datos originales caigan dentro de intervalos de una hora.
2. Analiza los datos.
3. Entrena diferentes modelos con diferentes hiperparámetros. La muestra de prueba debe ser el 10% del conjunto de datos inicial.
Prueba los datos usando la muestra de prueba y proporciona una conclusión.

Descripción de datos

Los datos se almacenan en el archivo /datasets/taxi.csv.  
El número de pedidos está en la columna num_orders.


Evaluación del proyecto
Hemos definido los criterios de evaluación para el proyecto. Léelos con atención antes de pasar al ejercicio.

Esto es en lo que se fijarán los revisores al examinar tu proyecto:

¿Seguiste todos los pasos de las instrucciones?
¿Cómo preparaste los datos?
¿Qué modelos e hiperparámetros consideraste?
¿Conseguiste evitar la duplicación del código?
¿Cuáles fueron tus hallazgos?
¿Mantuviste la estructura del proyecto?
¿Mantuviste el código limpio?
'''

# Importar librerías

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


'''
Paso 1: Cargar los datos y remuestrear a intervalos de una hora
'''


# Cargar los datos
file_path = '/datasets/taxi.csv'
data = pd.read_csv(file_path, parse_dates=['datetime'])


# Remuestrear los datos para que estén en intervalos de una hora
data = data.set_index('datetime').resample('H').sum().reset_index()


# Revisar los primeros registros para verificar el cambio
print(data.head())


'''
Paso 2: Análisis exploratorio de los datos
'''


# Resumen de los datos
print(data.describe())


# Distribución de pedidos por hora
plt.figure(figsize=(10, 5))
plt.plot(data['datetime'], data['num_orders'])
plt.title('Distribución de pedidos de taxis a lo largo del tiempo')
plt.xlabel('Fecha')
plt.ylabel('Número de pedidos')
plt.show()


'''
Paso 3: Separar el conjunto de datos en entrenamiento y prueba
'''

# Dividir los datos en características (X) y objetivo (y)
X = data[['datetime']]
y = data['num_orders']


# Convertir la característica 'datetime' en una característica más útil, como la hora
X['hour'] = X['datetime'].dt.hour
X = X[['hour']]  # Solo la hora como característica


# Separar los datos en entrenamiento y prueba (90% - 10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


'''
Paso 4: Entrenar modelos con diferentes hiperparámetros
'''


# Modelo de regresión lineal
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
print(f'Regresión Lineal RMSE: {lr_rmse:.2f}')

# Modelo de árbol de decisión
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
print(f'Árbol de Decisión RMSE: {dt_rmse:.2f}')

# Modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
print(f'Random Forest RMSE: {rf_rmse:.2f}') 