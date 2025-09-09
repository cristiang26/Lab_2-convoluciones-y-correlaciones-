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
### **- Estudiante 1.**
Representación de la convolución y gráfico secuencial a mano.

```python
h = np.array([5,6,0,0,8,4,2], dtype=int)
x = np.array([1,0,7,3,5,9,9,6,0,9], dtype=int)

y=np.convolve(h,x)

n_x=np.arange(len(x))
n_h=np.arange(len(h))
n_y=np.arange(len(y))
print("Marices inicales")
print("x:",x)
print("h:",h)
print("matriz convolucionada")
print("y:",y)

plt.figure()
plt.stem(n_y, y)
plt.title("convolución entre h y x")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.tight_layout()
plt.show()
```
Matriz convolucionada y: [ 5   6  35  57  51  79 157 136 102 143 172 102  42  84  36  18]

Gráfico realizado en python:\
<img width="780" height="575" alt="image" src="https://github.com/user-attachments/assets/1c0089eb-1cd5-480a-9772-225c6d5b9405" />\
Figura . Representación de la señal resultante en google colab

### **- Estudiante 2.**
Representación de la convolución y gráfico secuencial a mano.


```python
h_1 = np.array([5,6,0,0,8,7,9], dtype=int)
x_1 = np.array([1,0,7,5,8,7,0,2,9,5], dtype=int)

y=np.convolve(h_1,x_1)

n_x=np.arange(len(x_1))
n_h=np.arange(len(h_1))
n_y=np.arange(len(y))

print("Marices inicales")
print("x:",x_1)
print("h:",h_1)
print("matriz convolucionada")
print("y:",y)

plt.figure()
plt.stem(n_y, y)
plt.title("convolución entre h y x")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.tight_layout()
plt.show()
```
Matriz convolucionada y: [ 5   6  35  67  78  90 107  99 219 236 151  79  86 121 116  45]

Gráfico realizado en python:\
<img width="782" height="570" alt="image" src="https://github.com/user-attachments/assets/62d5857e-ac2c-40fe-b44c-0caaa3fb622c" />\
Figura . Representación de la señal resultante en google colab

### **- Estudiante 3.** 
Representación de la convolución y gráfico secuencial a mano.


```python
h_2 = np.array([5,6,0,0,9,1,1], dtype=int)
x_2 = np.array([1,0,1,4,1,8,0,0,0,7], dtype=int)

y=np.convolve(h_2,x_2)

n_x=np.arange(len(x_2))
n_h=np.arange(len(h_2))
n_y=np.arange(len(y))
print("Marices inicales")
print("x:",x_2)
print("h:",h_2)
print("matriz convolucionada")
print("y:",y)

plt.figure()
plt.stem(n_y, y)
plt.title("convolución entre h y x")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.tight_layout()
plt.show()
```
Matriz convolucionada y: [  5   6   5  26  38  47  58  37  14 112  51   8   0  63   7   7]

Gráfico realizado en python:\
<img width="781" height="572" alt="image" src="https://github.com/user-attachments/assets/066d9ecf-06e6-49e7-bb4b-7f4e41030c58" />\
Figura . Representación de la señal resultante en google colab

## **Parte B**
### Correlacion cruzada 

La correlación cruzada permite medir la similitud entre dos señales en función de un desplazamiento tempora, esta indicara cuanto se parecen las señales que tomamos, su resultado es util para poder indentificar patrones comunes, poder detectar coincidencias entre señales en procesamiento digital, para señales discretas, la correlacion cruzada esta definida por:\
<img width="317" height="92" alt="image" src="https://github.com/user-attachments/assets/645c4f67-b06b-4726-a23d-60c7e60185e3" />


Con las señales  x1[nTs]=cos⁡(2π100nTs)     para 0 ≤n< 9,   x2[nTs]=sin⁡(2π100nTs)     para 0 ≤n<  9 para Ts=1.25ms  
Se encuentra la correlación cruzada entre ambas señales con su respectiva representación gráfica.

```python
Ts = 1.25E-3
f = 100
N = 9
n = np.arange(N)
w0 = 2*np.pi*f*Ts
x1 = np.cos(w0*n)
x2 = np.sin(w0*n)

print("X1= ", np.round(x1, 4))
print("X2= ", np.round(x2, 4))

r12 = np.correlate(x1, x2, mode='full')
lags = np.arange(-(N-1), N)

plt.figure(figsize=(7, 4))
plt.stem(lags, r12, basefmt='r-')
plt.title("correlacion :)")
plt.xlabel("r12")
plt.ylabel("Correlacion entre X1 y X2")
plt.grid(True)
plt.show()
```

X1=  [ 1.      0.7071  0.     -0.7071 -1.     -0.7071 -0.      0.7071  1.    ]\
X2=  [ 0.      0.7071  1.      0.7071  0.     -0.7071 -1.     -0.7071 -0.    ]

<img width="766" height="491" alt="image" src="https://github.com/user-attachments/assets/1958bac0-62db-4766-ac9a-77089bafe0df" />\
Figura . Representación de la secuencia resultante en google colab
