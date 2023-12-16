import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fourier
import statistics as st


cnt = 0

# Genero 50 numeros utilizando una distribución normal estándar (media cero y varianza uno)
data = np.random.normal(0, 1, 50)
for i in range(len(data)):
    if data[i] <= 0:
        data[i] = 0
    elif data[i]>0 :
        data[i] = 1


# SOLO PARA TESTEAR
data = np.array([1., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0.])


fig, ax1 = plt.subplots()
ax1.stem(range(len(data)), data,use_line_collection=True)
ax1.grid()
ax1.set_xlabel('data')
ax1.set_ylabel('amplitude')
plt.title('Data before Transmiting')
plt.figure(1)

# Expreso data pero utilizando NRZ, es decir los 1 se expresan como 1 y los
# 0 se expresan como -1
data_NRZ = 2*data-1

# Realiza una conversion de serie a paralelo, esto lo hace expresando data_NRZ usando una
# matriz de 2 filas por 25 columnas (50/2)
s_p_data = np.zeros((2,int(len(data)/2)))

indexData = 0
for i in range(int(len(data)/2)):
    for j in range(2):
        s_p_data[j][i] = data_NRZ[indexData]
        indexData += 1


# Cantidad de bits que se transmiten por unidad de tiempo (en este caso Segundos CREO)
br = 1000000

# Frecuencia de la portadora mínima. 
f=br*10

# Representa la duración de un bit
T=1/br

# Crea un vector de 1 fila y 99 columnas con valores equiespaciados entre 0 y
# T (en este caso T = 1.0000e-06)
t = np.arange(T/99, T + T/99, T/99)

#  XXXXXXXXXXXXXXXXXXXXXXX QPSK modulation  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
y=np.array([])
y_in=np.array([])
y_qd=np.array([])
y_noise = np.array([])




for i in range(int(len(data)/2)):

    # y1_aux = s_p_data[0][i]
    # y2_aux = s_p_data[1][i]
    # print(y1_aux)
    # print(y2_aux)
    # print('--------------------')

    # inphase component
    y1 = s_p_data[0][i]*np.cos(2*np.pi*f*t)
    # Quadrature component
    y2 = s_p_data[1][i]*np.sin(2*np.pi*f*t)

    # inphase signal vector
    y_in = np.concatenate([y_in, y1])

    # print (y_in)

    # quadrature signal vector
    y_qd = np.concatenate([y_qd, y2])

    # print (y_qd)

    # randn(1,99) = Genero una matriz de 1x99 (1 fila y 99 columnas) que 
    # siguen una distribución normal con media cero y varianza uno.
    # Genero otra de la misma forma y sumo ambas, obteniendo  una matriz
    # 1x99 con sus valores sumados
    # Multiplico esto por la raiz cuadrada de 0.1
    
    # noise component
    # noise = ((sqrt(0.1)*(randn(1,99)+randn(1,99))));

    # part_1_noise = np.random.normal(0, 1, 99)

    # part_2_noise = np.random.normal(0, 1, 99)

    # part_3_noise = part_1_noise + part_2_noise



    #Originalmente era np.sqrt(0.1)

    #Con np.sqrt(15) se tiene un BER de 0.08 aprox

    noise = (np.sqrt(0.7) * (np.random.normal(0, 1, 99) + np.random.normal(0, 1, 99)))

    y_noise = np.concatenate([y_noise, noise])

    y = np.concatenate([y, (y1+y2+noise)])


# transmitting signal after modulation
Tx_sig=y

# Crea un vecto de 1 fila y 2475 columnas con valores equiespaciados
tt = np.arange(T/99, (T*len(data))/2 + T/99, T/99)

# Crear la figura y las subtramas
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
# Subtrama 1
axs[0].plot(tt, y_in)
axs[0].set_title('wave form for inphase component in QPSK modulation')
axs[0].set_xlabel('time(sec)')
axs[0].set_ylabel('amplitude(volt0)')
axs[0].grid(True)

# Subtrama 2
axs[1].plot(tt, y_qd)
axs[1].set_title('wave form for Quadrature component in QPSK modulation')
axs[1].set_xlabel('time(sec)')
axs[1].set_ylabel('amplitude(volt0)')
axs[1].grid(True)

# Subtrama 3
axs[2].plot(y_noise)
axs[2].set_title('wave form for noise component in QPSK modulation')
axs[2].set_xlabel('time(sec)')
axs[2].set_ylabel('amplitude(volt0)')
axs[2].grid(True)

# Subtrama 4
axs[3].plot(tt, Tx_sig, color='red')
axs[3].set_title('QPSK modulated signal (sum of inphase and Quadrature phase signal and noise)')
axs[3].set_xlabel('time(sec)')
axs[3].set_ylabel('amplitude(volt0)')
axs[3].grid(True)

# Ajustar diseño
plt.tight_layout()

plt.figure(2)

final_signal_with_noise = Tx_sig

# final_signal_with_noise = y_noise

var_len_input = 1238




# --/RECEPTOR/--


# plt.figure(2)

#Vector de frecuencias
F = np.arange(0, var_len_input)


plt.figure(3)

#Calculo de la transformada de Fourier (señal con ruido)
aux_noise = fourier.fft(final_signal_with_noise)
fourier_signal_whit_noise = abs(aux_noise)
plt.plot(F, fourier_signal_whit_noise[:var_len_input])
plt.xlabel('Frecuencia (Hz)[Señal con ruido]', fontsize='14')
plt.ylabel('Amplitud FFT ', fontsize='14')



plt.figure(4)

##Inicio de algoritmo LAD

#Arreglo con los valores de energía de la señal SUMANDO el ruido (se considera la mitad del espectro ya que se espejan los valores)
arreglo1 = fourier_signal_whit_noise[:var_len_input]**2


#Creamos un arreglo de tuplas. En el primer valor cada tupla tendrá el valor de la energia de la señal, en el segundo valor tendrá el indice en el espectro de frecuencia al que corresponde ese valor de energia.
index = np.arange(0, len(arreglo1))
arreglofinal = []
for i in range(0,len(arreglo1)):
  arreglofinal.append((arreglo1[i],index[i]))

#Ordenamos las muestras de manera creciente según su energia.
sorted_array = sorted(arreglofinal, key=lambda x: x[0])

plt.plot(index, [x[0] for x in sorted_array])
plt.yscale('log')
plt.xlabel('Espectro ordenado de manera creciente', fontsize='8')
plt.ylabel('Energia de la señal recibida', fontsize='14')

#Inicio de las iteraciones para encontrar umbrales
##En principio la elección de la pfa dependerá de que tan seguro quieras que sea la deleccion o decisión de presencia o no de la señal. Si tomas uno muy chico, termina optando por menos valores pero no le vas a errar.
##Tomar una PFA lo suficientemente grande para que no pierdas señal pero lo suficientemente chico para no tomar ruido como señal
pfa1 = 0.0001
pfa2 = 0.001

Tcme1 = -np.log(pfa1)
Tcme2 = -np.log(pfa2)

flag = 1
clean_set=sorted_array[1:round((0.01*len(sorted_array)))]; #Tomamos los 4 primeros elementos de sorted_array
#En realidad toma el 1% de los elementos partiendo desde el extremo de menor energia
#En este caso serian 5 elementos, pero al hacer 1:5, toma desde el indice 1 al 5, por lo tanto son 4 elementos
tu = Tcme1*st.mean([x[0] for x in clean_set])
tu_old = 0


#Condicion2_elem2
tuple_item_tu_old = []
for tuple_item in sorted_array:
    if tuple_item[0] < tu_old:
        tuple_item_tu_old.append(tuple_item)

len_tuple_item_tu_old = len(tuple_item_tu_old)

while(flag):

  #Condicion2_elem1
  tuple_item_tu = []
  for tuple_item in sorted_array:
      if tuple_item[0] < tu:
         tuple_item_tu.append(tuple_item)
  
  len_tuple_item_tu = len(tuple_item_tu)

  if tu == tu_old or len_tuple_item_tu <= len_tuple_item_tu_old:
    flag = 0
  else:
    len_tuple_item_tu_old = len_tuple_item_tu
    tu_old = tu
    tu = Tcme1*st.mean([x[0] for x in tuple_item_tu])


#Voy iterando el valor de la media hasta que llega un momento que:

#1-La media de la iteracion anterior y la actual son iguales, en este caso significa que los
#elementos de tuple_item_tu de la iteracion anterior y la actual son los mismos, por lo cual no
#tiene sentido seguir iterando ya que tuple_item_tu no va a variar.
#Luego me quedo con ese valor de la media como umbral superior

#2-El incremento de la media en la proxima iteracion considera la misma cantidad de elementos
#que antes (o menos) por lo tanto este incremento de la media no me aumenta (o hasta podria disminuir)
#el numero de elementos por debajo de la misma, entonces termina la iteracion y me quedo con ese 
#valor de la media como umbral superior




flag = 1
clean_set = sorted_array[1:round((0.01*len(sorted_array)))];
tl = Tcme2*st.mean([x[0] for x in clean_set])
tl_old = 0

#Condicion2_elem2
tuple_item_tl_old = []
for tuple_item in sorted_array:
    if tuple_item[0] < tl_old:
        tuple_item_tl_old.append(tuple_item)

len_tuple_item_tl_old = len(tuple_item_tl_old)

while(flag):

  #Condicion2_elem1
  tuple_item_tl = []
  for tuple_item in sorted_array:
      if tuple_item[0] < tl:
         tuple_item_tl.append(tuple_item)
  
  len_tuple_item_tl = len(tuple_item_tl)

  if tl == tl_old or len_tuple_item_tl <= len_tuple_item_tl_old:
    flag = 0
  else:
    len_tuple_item_tl_old = len_tuple_item_tl
    tl_old = tl
    tl = Tcme2*st.mean([x[0] for x in tuple_item_tl])



print("Umbral superior: ")
print(tu)
print("Umbral inferior:")
print(tl)


plt.figure(5)


plt.plot(index, [x[0] for x in sorted_array])
plt.axhline(y=tu, color='red', linestyle='--',linewidth=0.5)
plt.axhline(y=tl, color='green', linestyle='--',linewidth=0.5)
plt.yscale('log')
plt.xlabel('Espectro ordenado de manera creciente', fontsize='8')
plt.ylabel('Energia de la señal recibida', fontsize='14')


plt.figure(6)

#Del arreglo de muestras ordenado de manera creciente segun su energía, solo consideramos
#aquellos cuya energía es mayor al umbral inferior
muestras_finales_aux = [tuple_item for tuple_item in sorted_array if tl <= tuple_item[0]]

#Del arreglo anterior ahora solo considero aquellos cuya energía es mayor al umbral superior
muestras_finales = [tuple_item for tuple_item in muestras_finales_aux if tu <= tuple_item[0]]

#print(len(muestras_finales))

#Extraigo el indice de cada una de las muestras cuya energía es mayor al umbral superior (por ende tambien mayor al umbral inferior)
index_final_with_signal = [tuple_item[1] for tuple_item in muestras_finales]


#Considerando la transformada de fourirer de la señal CON RUIDO
plt.plot(F, fourier_signal_whit_noise[:var_len_input])

for index in range(0,len(index_final_with_signal)):
  plt.scatter(index_final_with_signal[index], 400, color='red', marker='o')
plt.xlabel('Frecuencia (Hz)', fontsize='14')
plt.ylabel('Amplitud FFT', fontsize='14')

print('Cantidad de puntos rojos: ')
print(len(index_final_with_signal))





#  XXXXXXXXXXXXXXXXXXXXXXXXXXXX QPSK demodulation XXXXXXXXXXXXXXXXXXXXXXXXXX

Rx_data = np.array([])
# Received signal
# Rx_sig = Tx_sig
Rx_sig = final_signal_with_noise
rx_qd_data = np.array([])

for i in range(0, len(Rx_sig), 99):
    # XXXXXX inphase coherent dector XXXXXXX
    # Es un algoritmo que lo que hace es que para cada iteracion va tomando 
    # los 99 elementos siguientes de la señal recibida Rx_sig, esto para 
    # obtener los valores de cada uno de los simbolos lo cual lo hace 
    # a medida que van pasando cada una a las iteraciones
    # Luego multiplica por el cos
    # Z_in=Rx_sig((i-1)*length(t)+1:i*length(t)).*cos(2*pi*f*t);
    Z_in_aux = Rx_sig[i:i + 99]

    Z_in = Z_in_aux*np.cos(2*np.pi*f*t)
    # Hace la sumatoria de los valores del simbolo multiplicado
    # por el cos pero el tipo de sumatoria es de intergacion
    # trapezoidal
    Z_in_intg = (np.trapz(Z_in, t))*(2/T)
    if Z_in_intg > 0:
        Rx_in_data = 1
    else:
        Rx_in_data = 0


    # XXXXXX Quadrature coherent dector XXXXXX
    # Lo mismo que el anterior pero multiplica por el sen
    # Z_qd=Rx_sig((i-1)*length(t)+1:i*length(t)).*sin(2*pi*f*t);
    Z_qd = Rx_sig[i:i + 99]*np.sin(2*np.pi*f*t)

    # Lo mismo que el anterior
    Z_qd_intg = (np.trapz(Z_qd, t))*(2/T)
    if Z_qd_intg > 0:
        Rx_qd_data = 1
    else:
        Rx_qd_data = 0

    # Rx_data = np.concatenate([Rx_data, np.concatenate([Rx_in_data, Rx_qd_data])])
    Rx_data = np.append(Rx_data, Rx_in_data)
    Rx_data = np.append(Rx_data, Rx_qd_data)

    rx_qd_data = np.append(rx_qd_data, Rx_qd_data)

    
# Algortimo para calcular y mostrar el BER (bit error probability)
# (probabilidad de error de bit) (La tasa de errores de bits )

cnt = 0

for i in range(len(Rx_data)):
    if data[i] != Rx_data[i]:
        cnt += 1


fig, ax2 = plt.subplots()
ax2.stem(range(len(Rx_data)), Rx_data,use_line_collection=True)
ax2.grid()
ax2.set_xlabel('data')
ax2.set_ylabel('amplitude')
plt.title('Information after Receiveing')
plt.figure(7)

print('bit error probabilty is (BER): ')
print(cnt/(int(len(Rx_data))))




plt.show()
