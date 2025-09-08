# Lab_2-Convoluciones-y-correlaciones-
En esta practica se analizó la convolución como una operación que permite obtener la respuesta de un sistema discreto ante una entrada determinada, así mismo se aplicó la transformada de Fourier a la señal obtenida con el microcontrolador. \
Las librerias que se utilizaron fueron las siguientes:
```python
import numpy as np
import matplotlib.pyplot as plt
!pip install wfdb   #Instalacion en colab
import wfdb
import pandas as pd
```
numpy: Librería para cálculos numéricos y manejo de arreglos (vectores y matrices).
matplotlib.pyplot: Sirve para generar gráficos y visualizar datos.
wfdb: Librería especializada para leer y manipular señales biomédicas del formato PhysioNet, como ECG.
pandas: Manejo de datos en forma de tablas (aunque aquí solo se importa, no se usa mucho).\

Con esto se realiza un diagrama de flujo para cada parte de este laboratorio 
## **Parte A**
En esta parte se encontró la señal resultante y[n] de la convolución del sistema h[n]= {codigo de cada estudiante} y la señal x[n]={cada digito de la cedula}. Así tambien se realizó una representación gráfica y secuencial de la señal encontrada a mano y python.
