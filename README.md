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
**Convolución**

Esta describe como una señal de entrada se transfomra al pasar por un sistema. En procesamiento de señales, se utiliza mas que todo para calculas la respuesta de un sistema lineal en el tiempo cuando conocemos su entrada y su respuesta de impulso. \
Se puede definir como: 
<img width="326" height="84" alt="image" src="https://github.com/user-attachments/assets/5b8c9086-0613-457a-bb93-ee05b02d3d8c" /> \ 

Donde: 

+ x[n] = señal de entrada
+ h[n] = respuesta al impulso del sistema
+ y[n] = salida del sistema 

En esta parte se encontró la señal resultante y[n] de la convolución del sistema h[n]= {codigo de cada estudiante} y la señal x[n]={cada digito de la cedula}. Así tambien se realizó una representación gráfica y secuencial de la señal encontrada a mano y python.

### **- Estudiante 1.**
Representación de la convolución y gráfico secuencial a mano.

<img width="574" height="566" alt="image" src="https://github.com/user-attachments/assets/18fe023d-d1ee-4c26-9891-b8f9224fad0f" />

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

<img width="468" height="556" alt="image" src="https://github.com/user-attachments/assets/1cd81ca2-04a7-4aa4-8838-52ef04dac61b" />


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

La correlación cruzada permite medir la similitud entre dos señales en función de un desplazamiento tempora, esta indicara cuanto se parecen las señales que tomamos, su resultado es util para poder indentificar patrones comunes, poder detectar coincidencias entre señales en procesamiento digital, para señales discretas, la correlacion cruzada esta definida por: \
<img width="317" height="92" alt="image" src="https://github.com/user-attachments/assets/645c4f67-b06b-4726-a23d-60c7e60185e3" /> 

Donde: 
+ x[n] y Y[n]: las dos señales tomadas.

+ K: Desplazamineto

+ r: similitud cuanto se dezplasa y[n] en K

Con las señales  x1[nTs]=cos⁡(2π100nTs)     para 0 ≤n< 9,   x2[nTs]=sin⁡(2π100nTs)     para 0 ≤n<  9 para Ts=1.25ms  
Se encuentra la correlación cruzada entre ambas señales con su respectiva representación gráfica.

**Diagrama de flujo parte B**

<img width="314" height="832" alt="image" src="https://github.com/user-attachments/assets/46d12bd5-b0c0-481f-8da1-ac5d5249646c" />


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

## **Parte C**

En esta parte se analizó una señal muestreada a 2000 Hz tanto en el dominio del tiempo como en el de la frecuencia. Primero, se graficó la señal en función del tiempo y se calcularon sus principales estadísticos descriptivos (media, mediana, desviación estándar, valor máximo y mínimo), lo que permite caracterizar su comportamiento general y amplitud. Luego, se aplicó la Transformada Rápida de Fourier (FFT), que es un algoritmo eficiente para calcular la Transformada Discreta de Fourier y permite descomponer la señal en sus componentes de frecuencia, mostrando qué frecuencias están presentes y con qué magnitud. A continuación, se estimó la densidad espectral de potencia, que describe cómo se distribuye la energía de la señal a lo largo del espectro de frecuencias. Para calcularla se utilizó el método de Welch, el cual divide la señal en segmentos superpuestos, les aplica una ventana de suavizado y promedia los periodogramas resultantes; de esta manera se obtiene una estimación más robusta y menos ruidosa de la energía espectral, lo que facilita la interpretación y el análisis de las frecuencias dominantes en la señal.



**Diagrama de flujo parte C**
<img width="551" height="768" alt="image" src="https://github.com/user-attachments/assets/05d711b7-0aa0-47a0-8d1f-c11b61a55cc4" />

Por medio del generador de señales se logro crear una señal de electrooculometria a una frecuancia de 1000Hz para poder tomar todos los valores requeridos para este laboratorio, por medio del DAQ y un codigo suministrado por la docente se logro capturar y descargar correctamente la señal que generamos anteriormente.

El siguiente codigo fue el suministrado por la docente, este codigo es con el fin de poder capturar la señal que creamos con el generador de señales, y por medio del DAQ se logro capturar la señal requerida para su posterior procesamiento y analisis, para asi poder lograr con los objetivos del laboratorio.

```python
# -*- coding: utf-8 -*-
"""
Script para captura de señales usando la DAQ.
Este script permite configurar la frecuencia de muestreo y duración de la señal;
realiza la captura internamente y posteriormente grafica la señal completa. 
Es una captura eficiente y exacta en el tiempo. Es útil si quiero grabar la señal
para procesarla posteriormente usando otro script, lenguaje, entorno, etc.

Se requiere instalar: 
Librería de uso de la DAQ
!python -m pip install nidaqmx     

Driver NI DAQ mx
!python -m nidaqmx installdriver   

Created on Thu Aug 21 08:36:05 2025
@author: Carolina Corredor
"""

# Librerías: 
import nidaqmx                     # Librería daq. Requiere haber instalado el driver nidaqmx
from nidaqmx.constants import AcquisitionType # Para definir que adquiera datos de manera consecutiva
import matplotlib.pyplot as plt    # Librería para graficar
import numpy as np                 # Librería de funciones matemáticas

#%% Adquisición de la señal por tiempo definido

fs = 2000           # Frecuencia de muestreo en Hz. Recordar cumplir el criterio de Nyquist
duracion = 5      # Periodo por el cual desea medir en segundos
senal = []          # Vector vacío en el que se guardará la señal
dispositivo = 'Dev1/ai0' # Nombre del dispositivo/canal (se puede cambiar el nombre en NI max)

total_muestras = int(fs * duracion)

with nidaqmx.Task() as task:
    # Configuración del canal
    task.ai_channels.add_ai_voltage_chan(dispositivo)
    # Configuración del reloj de muestreo
    task.timing.cfg_samp_clk_timing(
        fs,
        sample_mode=AcquisitionType.FINITE,   # Adquisición finita
        samps_per_chan=total_muestras        # Total de muestras que quiero
    )

    # Lectura de todas las muestras de una vez
    senal = task.read(number_of_samples_per_channel=total_muestras)

t = np.arange(len(senal))/fs # Crea el vector de tiempo 
plt.plot(t,senal)
plt.axis([0,duracion,-10,10])
plt.grid()
plt.ylim(-2.5 , 2.5)
plt.xlim(0 , 0.2)
plt.title(f"fs={fs}Hz, duración={duracion}s, muestras={len(senal)}")
plt.show()

np.savetxt(f"señal_fs{fs}_t{duracion}_2.txt", senal)
```

Ya con la señal descargada en un archivo .txt se logro graficar por medio de codigo Python, y asi con esto poder seguir con los siguientes incisos requeridos.

```python
fN = 800
senal = np.loadtxt("/content/drive/MyDrive/señal_fs2000_t5.txt")


t = np.arange(len(senal)) / fN
plt.figure(figsize=(10,5))
plt.plot(t, senal)
plt.xlabel("Tiempo (s)")
plt.ylabel(" Voltage (mV)")
plt.title("Señal capturada")
plt.grid(True)
plt.show()
```
<img width="768" height="414" alt="image" src="https://github.com/user-attachments/assets/62b81b90-9be3-433a-8c98-9bfffe4abc52" />

**Estadisticos en el dominio del timepo**

Se caracterizo la señal por los diferentes estadisticos en el tiempo, ya vendria siendo la media, mediana, deviacion estandar, maximo y minimo de la señal que capturamos anteriormente.
```python
# Estadísticos en tiempo
print("Media:", np.mean(senal))
print("Mediana:", np.median(senal))
print("Desviación estándar:", np.std(senal))
print("Máximo:", np.max(senal))
print("Mínimo:", np.min(senal))
```
+ Media: -0.14938

+ Mediana: -0.0791

+ Desviación estándar: 0.3927

+ Máximo: 1.4549

+ Mínimo: -1.53142

La señal biologica adquirida en el laboratorio por medio del generador de señales, esta se puede clasificar como aleatoria, aperiodica y digital, ya que proviene de un "fenomeno" que no es completamente predecible a futuro, no presenta una repeticion exacta en el timepo y podemos decir que es digital, pues originalmente es analogica al ser capturada por el DAQ y procesada por Python se convierte en una señal digital representada por valores discretos.

**Transformada de fourier**

```python
N = len(senal)
fft_v = np.fft.fft(senal)
fft_Fre = np.fft.fftfreq(N, 1/fN)

fft_magnitud = np.abs(fft_v[:N//2])
frecuencias = fft_Fre[:N//2]

espectro = fft_magnitud / np.sum(fft_magnitud)

plt.figure(figsize=(10,5))
plt.plot(fft_Fre, np.abs(fft_v))
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.title("Transformada de Fourier")
```
<img width="802" height="402" alt="image" src="https://github.com/user-attachments/assets/e2093942-603b-481c-b5cf-d91af709e444" />

**Densidad espectral**
```python
# Densidad espectral de potencia (Welch)
pxx, f_welch = welch(senal, fs, nperseg=min(1024, N))
plt.figure(figsize=(10,4))
plt.semilogy(f_welch, pxx)
plt.title("Densidad espectral de potencia")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.grid(True)
plt.show()
```
<img width="785" height="352" alt="image" src="https://github.com/user-attachments/assets/c099dd22-8069-422e-ad9e-ff3eba89ce54" />
**Estadisticos en el dominio de la frecuencia**

**Frecuencia media**
```python
f_media = np.sum(frecuencias * espectro)
print(f"Frecuencia media: {f_media} Hz")
```
Frecunecia media  : 102.2935 Hz

**Frecuancia mediana**
```python
f_acumulada = np.cumsum(espectro)
f_mediana = frecuencias[np.argmin(np.abs(f_acumulada - 0.5))]
print(f"Frecuencia mediana: {f_mediana} Hz")
```
Frecuencia mediana: 55.92 Hz

**Desviacion estandar**
```python
f_varianza = np.sum(((frecuencias - f_media)**2) * espectro)
f_desviacion_estandar = np.sqrt(f_varianza)
print(f"Desviación estándar en frecuencia: {f_desviacion_estandar} Hz")
```
Desviacion estandar en frecuancia: 110.9863 Hz

**Histogramas de frecuancias**
```python
plt.figure(figsize=(8,5))
plt.hist(frecuencias, bins=30, weights=espectro, color='blue')
plt.title("Histograma de frecuencias")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de probabilidad")
plt.grid(True)
plt.show()
```
<img width="772" height="513" alt="image" src="https://github.com/user-attachments/assets/311db35a-6768-468d-aaf5-fed35cf6a82f" />




