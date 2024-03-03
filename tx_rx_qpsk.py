import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fourier
import statistics as st

# --/DEFINICION DE VARIABLES/--

# Cantidad de bits a transmitir
cant_bits_t = 50

# Cantidad de bits que se transmiten por unidad de tiempo (en este caso Segundos CREO)
br = 1000000

# Frecuencia de la portadora mínima. 
f=br*10

# Representa la duración de un bit
T=1/br

# Cantidad de valores de las abscisas para un simbolo
num_absc = 99

# Valor para la raiz cuadrada en la generacion del ruido
sqrt_noise = 0.7

# LAD

# En principio la elección de la pfa dependerá de que tan seguro quieras que sea la deteccion o decisión de 
# presencia o no de la señal. Si tomas uno muy chico, termina optando por menos valores pero no le vas a errar.
# Tomar una PFA lo suficientemente grande para que no pierdas señal pero lo suficientemente chico para no tomar ruido como señal
pfa1 = 0.0001
pfa2 = 0.001

# Porcentaje de elementos para comenzar algoritmo LAD
percent_elements = 0.01

# Numero de muestras de distancia entre dos clusters para considerar que los mismos consisten en la misma señal
num_samples = 2

# --/DESARROLLO DEL ALGORITMO/--


# Genero 50 numeros utilizando una distribución normal estándar (media cero y varianza uno)
# Estos van a ser los 50 bits que se van a tener a la entrada del transmisor y que debe transmitir
data = np.random.normal(0, 1, cant_bits_t)
for i in range(len(data)):
    if data[i] <= 0:
        data[i] = 0
    elif data[i]>0 :
        data[i] = 1

# SOLO PARA TESTEAR
# Conjunto de 50 bits de entrada del transmisor para hacer pruebas
data = np.array([1., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0.])

# Plot de los bits a transmitir
fig, bits_t = plt.subplots()
bits_t.stem(range(len(data)), data,use_line_collection=True)
bits_t.grid()
bits_t.set_xlabel('data')
bits_t.set_ylabel('amplitude')
plt.title('Data before Transmiting')
plt.figure(1)

# Expreso data pero utilizando NRZ, es decir los 1 se expresan como 1 y los
# 0 se expresan como -1
data_NRZ = 2*data-1

# Realiza una conversion de serie a paralelo, esto lo hace expresando data_NRZ usando una
# matriz de 2 filas por 25 columnas (50/2)
# De esta forma cada columna representa el simbolo a transmitir, formado por 2 bits cada
# simbolo, ya que se esta transmitiendo una señal QPSK
s_p_data = np.zeros((2,int(len(data)/2)))

indexData = 0
for i in range(int(len(data)/2)):
    for j in range(2):
        s_p_data[j][i] = data_NRZ[indexData]
        indexData += 1


# Crea un vector de 1 fila y 99 columnas con valores equiespaciados entre 0 y
# T (en este caso T = 1.0000e-06)
# Que representa los valores de las abscisas para un simbolo
t = np.arange(T/num_absc, T + T/num_absc, T/num_absc)

#  XXXXXXXXXXXXXXXXXXXXXXX QPSK modulation  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
y=np.array([])
y_in=np.array([])
y_qd=np.array([])
y_noise = np.array([])

for i in range(int(len(data)/2)):
    # Inphase component
    y1 = s_p_data[0][i]*np.cos(2*np.pi*f*t)
    # Quadrature component
    y2 = s_p_data[1][i]*np.sin(2*np.pi*f*t)

    # Inphase signal vector
    y_in = np.concatenate([y_in, y1])
    # Quadrature signal vector
    y_qd = np.concatenate([y_qd, y2])

    # Generacion de la componente de ruido
    # Genero una matriz de 1x99 (1 fila y 99 columnas) que 
    # siguen una distribución normal con media cero y varianza uno.
    # Genero otra de la misma forma y sumo ambas, obteniendo una matriz
    # 1x99 con sus valores sumados
    # Multiplico esto por la raiz cuadrada de 0.1
    
    #Originalmente era np.sqrt(0.1)
    #Con np.sqrt(15) se tiene un BER de 0.08 aproximadamente

    noise = (np.sqrt(sqrt_noise) * (np.random.normal(0, 1, num_absc) + np.random.normal(0, 1, num_absc)))

    y_noise = np.concatenate([y_noise, noise])

    y = np.concatenate([y, (y1+y2+noise)])


# Señal a transmitir luego de la modulacion
Tx_sig=y

# Crea un vector de 1 fila y 2475 columnas con valores equiespaciados
# Que representa los valores de las abscisas para toda la señal a transmitir
tt = np.arange(T/num_absc, (T*len(data))/2 + T/num_absc, T/num_absc)


fig, wave_t = plt.subplots(4, 1, figsize=(10, 8))
# Subtrama 1: Señal de Inphase component
wave_t[0].plot(tt, y_in)
wave_t[0].set_title('wave form for inphase component in QPSK modulation')
wave_t[0].set_xlabel('time(sec)')
wave_t[0].set_ylabel('amplitude(volt0)')
wave_t[0].grid(True)

# Subtrama 2: Señal de Quadrature component
wave_t[1].plot(tt, y_qd)
wave_t[1].set_title('wave form for Quadrature component in QPSK modulation')
wave_t[1].set_xlabel('time(sec)')
wave_t[1].set_ylabel('amplitude(volt0)')
wave_t[1].grid(True)

# Subtrama 3: Señal de ruido
wave_t[2].plot(y_noise)
wave_t[2].set_title('wave form for noise component in QPSK modulation')
wave_t[2].set_xlabel('time(sec)')
wave_t[2].set_ylabel('amplitude(volt0)')
wave_t[2].grid(True)

# Subtrama 4: Señal final a transmitir
wave_t[3].plot(tt, Tx_sig, color='red')
wave_t[3].set_title('QPSK modulated signal (sum of inphase and Quadrature phase signal and noise)')
wave_t[3].set_xlabel('time(sec)')
wave_t[3].set_ylabel('amplitude(volt0)')
wave_t[3].grid(True)

# Ajustar diseño
plt.tight_layout()
plt.figure(2)

# Señal final a transmitir Simbolos + Ruido
final_signal_with_noise = Tx_sig

# SOLO PARA TESTEAR
# Señal final a transmitir SOLO ruido
# final_signal_with_noise = y_noise

# Este valor es la mitad de 2475, es decir 1237.5 pero lo redondeo a 1238
# donde 2475 corresponde al intervalo de abscisas de la señal a transmitir
# Se considera la mitad ya que cuando se analiza el espectro los valores se espejan
var_len_input = round(len(tt)/2)




# --/RECEPTOR/--


# Valores de abscisas para la transformada de Fourier
F = np.arange(0, var_len_input)


#Calculo de la transformada de Fourier de la señal recibida
aux_noise = fourier.fft(final_signal_with_noise)
fourier_signal_whit_noise = abs(aux_noise)
fig, fft_r = plt.subplots()
fft_r.plot(F, fourier_signal_whit_noise[:var_len_input])
fft_r.set_xlabel('Frecuencia (Hz)[Señal con ruido]', fontsize='14')
fft_r.set_ylabel('Amplitud FFT ', fontsize='14')
plt.title('Transformada de Fourier de la señal recibida')
plt.figure(3)


## Inicio de algoritmo LAD

# Arreglo con los valores de energía de la señal SUMANDO el ruido 
# (se considera la mitad del espectro ya que se espejan los valores)
energy_of_signal_r = fourier_signal_whit_noise[:var_len_input]**2

# Creamos un arreglo de tuplas. En el primer valor cada tupla tendrá el valor de la energia de la señal, 
# en el segundo valor tendrá el indice en el espectro de frecuencia al que corresponde ese valor de energia.
index = np.arange(0, len(energy_of_signal_r))
arreglofinal = []
for i in range(0,len(energy_of_signal_r)):
  arreglofinal.append((energy_of_signal_r[i],index[i]))

# Ordenamos las muestras de manera creciente según su energia.
sorted_array = sorted(arreglofinal, key=lambda x: x[0])

fig, energy_s = plt.subplots()
energy_s.plot(index, [x[0] for x in sorted_array])
energy_s.set_yscale('log')
energy_s.set_xlabel('Espectro ordenado de manera creciente', fontsize='8')
energy_s.set_ylabel('Energia de la señal recibida', fontsize='14')
plt.title('Energia ordenada de manera creciente')
plt.figure(4)

# Inicio de las iteraciones para encontrar umbrales

Tcme1 = -np.log(pfa1)
Tcme2 = -np.log(pfa2)

flag = 1
clean_set=sorted_array[0:round((percent_elements*len(sorted_array)))]
# Toma el 1% de los elementos partiendo desde el extremo de menor energia
# En este caso serian 12 elementos, ya que toma los valores de los indices del 0:11 (con el 11 incluido)
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
clean_set = sorted_array[0:round((percent_elements*len(sorted_array)))]
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

fig, energy_s_t = plt.subplots()
energy_s_t.plot(index, [x[0] for x in sorted_array])
energy_s_t.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
energy_s_t.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
energy_s_t.set_yscale('log')
energy_s_t.set_xlabel('Espectro ordenado de manera creciente', fontsize='8')
energy_s_t.set_ylabel('Energia de la señal recibida', fontsize='14')
plt.title('Energia ordenada de manera creciente con umbrales')
plt.figure(5)


fig, fft_r_lad_t = plt.subplots()
fft_r_lad_t.plot(F, energy_of_signal_r)
fft_r_lad_t.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
fft_r_lad_t.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
fft_r_lad_t.set_xlabel('Frecuencia (Hz)', fontsize='14')
fft_r_lad_t.set_ylabel('Energia de la señal recibida', fontsize='14')
plt.title('Energia de la señal recibida con umbrales del LAD')
plt.figure(6)


cluster_aux = []

flag_saving_cluster = False

flag_saving_cluster_first_element = True

flag_cluster_signal = False

clusters_list = []

clusters_list_signal = []

for i in range(0,len(arreglofinal)):
   
   if arreglofinal[i][0] >= tl:
        flag_saving_cluster = True
        if (flag_saving_cluster_first_element == True):
           if(0 < i):
              cluster_aux.append(arreglofinal[i-1])
           flag_saving_cluster_first_element = False
        cluster_aux.append(arreglofinal[i])
        if arreglofinal[i][0] >= tu:
           flag_cluster_signal = True
        
   if((arreglofinal[i][0] < tl and flag_saving_cluster == True) or (i == (len(arreglofinal)-1) and flag_saving_cluster == True)):
      flag_saving_cluster = False
      flag_saving_cluster_first_element = True
      cluster_aux.append(arreglofinal[i])
      clusters_list.append(cluster_aux)
      if flag_cluster_signal == True:
         clusters_list_signal.append(cluster_aux)
      flag_cluster_signal = False
      cluster_aux = []

# BORRAR    
print(clusters_list)

# BORRAR
print(clusters_list_signal)

signal_aux = []

signal_list = []

# Tomo el primer cluster y lo agrego como la primer señal

for i in range(0,len(clusters_list_signal[0])):
   signal_aux.append(clusters_list_signal[0][i])

signal_list.append(signal_aux)

signal_aux = []

# Luego empiezo a comparar la ultima señal agregada a signal_list con los clusters siguientes 
# Por eso comienza en 1

for i in range(1,len(clusters_list_signal)):
   
   index_last_signal = len(signal_list)-1
   index_last_sample_last_signal = len(signal_list[index_last_signal])-1
   sample_id_last_sample_last_signal = signal_list[index_last_signal][index_last_sample_last_signal][1]

   sample_id_first_sample_current_cluster = clusters_list_signal[i][0][1]

   for j in range(0,len(clusters_list_signal[i])):
      signal_aux.append(clusters_list_signal[i][j])

   if((sample_id_first_sample_current_cluster - sample_id_last_sample_last_signal) <= num_samples):
      
      index_start_signal_aux = 0
      if((sample_id_first_sample_current_cluster - sample_id_last_sample_last_signal) == 0):
         index_start_signal_aux = 1

      for j in range(index_start_signal_aux,len(signal_aux)):
         signal_list[index_last_signal].append(signal_aux[j])
      
   else:
      
      signal_list.append(signal_aux)
      
   signal_aux = []


# BORRAR
print(signal_list)
      

fig, signals_by_lad = plt.subplots()
signals_by_lad.plot(F, energy_of_signal_r)
for i in range(0,len(signal_list)):
    signals_by_lad.plot([x[1] for x in signal_list[i]], [y[0] for y in signal_list[i]])
signals_by_lad.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
signals_by_lad.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
signals_by_lad.set_xlabel('Frecuencia (Hz)', fontsize='14')
signals_by_lad.set_ylabel('Energia de la señal recibida', fontsize='14')
plt.title('Señales identificadas mediante metodo LAD')
plt.figure(7)



# Del arreglo de muestras ordenado de manera creciente segun su energía, solo consideramos
# aquellos cuya energía es mayor al umbral inferior
muestras_finales_aux = [tuple_item for tuple_item in sorted_array if tl <= tuple_item[0]]

# Del arreglo anterior ahora solo considero aquellos cuya energía es mayor al umbral superior
muestras_finales = [tuple_item for tuple_item in muestras_finales_aux if tu <= tuple_item[0]]

# Extraigo el indice de cada una de las muestras cuya energía es mayor al umbral superior (por ende tambien mayor al umbral inferior)
index_final_with_signal = [tuple_item[1] for tuple_item in muestras_finales]

# Considerando la transformada de fourirer de la señal CON RUIDO
fig, fft_r_lad = plt.subplots()
fft_r_lad.plot(F, fourier_signal_whit_noise[:var_len_input])

for index in range(0,len(index_final_with_signal)):
  fft_r_lad.scatter(index_final_with_signal[index], 400, color='red', marker='o')
fft_r_lad.set_xlabel('Frecuencia (Hz)', fontsize='14')
fft_r_lad.set_ylabel('Amplitud FFT', fontsize='14')
plt.title('Transformada de Fourier de la señal recibida mas metodo LAD')
plt.figure(8)

print('Cantidad de puntos rojos: ')
print(len(index_final_with_signal))




#  XXXXXXXXXXXXXXXXXXXXXXXXXXXX QPSK demodulation XXXXXXXXXXXXXXXXXXXXXXXXXX

Rx_data = np.array([])
# Señal recibida
Rx_sig = final_signal_with_noise

for i in range(0, len(Rx_sig), num_absc):
    # XXXXXX inphase coherent dector XXXXXXX
    # Es un algoritmo que lo que hace es que para cada iteracion va tomando 
    # los 99 elementos siguientes de la señal recibida Rx_sig, esto para 
    # obtener los valores de cada uno de los simbolos lo cual lo hace 
    # a medida que van pasando cada una de las iteraciones
    # Luego multiplica por el cos
    Z_in_aux = Rx_sig[i:i + num_absc]

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
    Z_qd = Rx_sig[i:i + num_absc]*np.sin(2*np.pi*f*t)

    # Lo mismo que el anterior
    Z_qd_intg = (np.trapz(Z_qd, t))*(2/T)
    if Z_qd_intg > 0:
        Rx_qd_data = 1
    else:
        Rx_qd_data = 0

    Rx_data = np.append(Rx_data, Rx_in_data)
    Rx_data = np.append(Rx_data, Rx_qd_data)


# Algortimo para calcular y mostrar el BER (bit error probability)
# (probabilidad de error de bit) (La tasa de errores de bits )

# Contador de bits recibidos de manera incorrecta
cnt = 0

for i in range(len(Rx_data)):
    if data[i] != Rx_data[i]:
        cnt += 1

fig, bits_r = plt.subplots()
bits_r.stem(range(len(Rx_data)), Rx_data,use_line_collection=True)
bits_r.grid()
bits_r.set_xlabel('data')
bits_r.set_ylabel('amplitude')
plt.title('Information after Receiveing')
plt.figure(9)

print('bit error probabilty is (BER): ')
print(cnt/(int(len(Rx_data))))

plt.show()
