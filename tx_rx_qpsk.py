import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fourier
import statistics as st
from numpy.fft import fft, fftshift
from scipy.integrate import simps
from math import pi
from numpy import cos, absolute

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

# Ventana de frecuencias a ver
cota_inferior_frec = 0

cota_superior_frec = 600

# LAD

# En principio la elección de la pfa dependerá de que tan seguro quieras que sea la deteccion o decisión de 
# presencia o no de la señal. Si tomas uno muy chico, termina optando por menos valores pero no le vas a errar.
# Tomar una PFA lo suficientemente grande para que no pierdas señal pero lo suficientemente chico para no tomar ruido como señal
pfa1 = 0.0001
pfa2 = 0.001

# Porcentaje de elementos para comenzar algoritmo LAD
percent_elements = 0.01

# Numero de muestras de distancia entre dos clusters para considerar que los mismos consisten en la misma señal
num_samples = 10

# --/DESARROLLO DEL ALGORITMO/--


# --/FM Signal/--

# Muestras por segundo que se van a tomar de la señal analogica
fs = 8000 # Sampling Frequency [Muestras/Seg]

# Parametros Modulante 1
fm = 5 # Message Frequency
Am = 1 # Message Amplitude

# Parametros Modulante 2
fm_2 = 5 # Message Frequency
Am_2 = 1 # Message Amplitude

# Parametros Portadora 1
fc = 200 # Carrier Frequency
Ac = 1 # Carrier Amplitude

# Parametros Portadora 2
fc_2 = 400 # Carrier Frequency
Ac_2 = 1 # Carrier Amplitude

kf = 100*pi # Modulation Coefficient [Rad/Volt*Seg]


# 8000 valores equiespaciados por 0.000125
tt = np.arange(0,1,1/fs)


# Señal Modulante 1
M = Am*cos(2*pi*fm*tt) # Message Signal

# Señal Modulante 1
M_2 = Am_2*cos(2*pi*fm_2*tt) # Message Signal

#Señal Portadora 1
C = Ac*cos(2*pi*fc*tt) # Carrier Signal

#Señal Portadora 2
C_2 = Ac_2*cos(2*pi*fc_2*tt) # Carrier Signal


# Integral de la Modulante 1
# Integración numérica de la señal del mensaje en función del tiempo utilizando la regla de Simpson
M_int = [0,]
for i in range(1,len(tt)):
    M_int.append(simps(M[:i],tt[:i]))

# Integral de la Modulante 2
# Integración numérica de la señal del mensaje en función del tiempo utilizando la regla de Simpson
M_int_2 = [0,]
for i in range(1,len(tt)):
    M_int_2.append(simps(M_2[:i],tt[:i]))

# Señal a transmitir SIN ruido. Contiene las portadoras moduladas
s_FM = []
for (ti, m, m_2) in zip(tt, M_int, M_int_2):
    # s_FM.append(Ac*cos(2*pi*fc*ti + kf*m))
    s_FM.append((Ac*cos(2*pi*fc*ti + kf*m))+(Ac_2*cos(2*pi*fc_2*ti + kf*m_2)))
    
# Señal de ruido
noise = (np.sqrt(0.7) * (np.random.normal(0, 1, len(s_FM)) + np.random.normal(0, 1, len(s_FM))))


# Señal a transmitir CON ruido
s_FM_noise =[]
s_FM_noise = s_FM + noise

# --/FM Signal/--

# Crea un vector de 1 fila y 2475 columnas con valores equiespaciados
# Que representa los valores de las abscisas para toda la señal a transmitir
# tt = np.arange(T/num_absc, (T*len(data))/2 + T/num_absc, T/num_absc)


fig, wave_t = plt.subplots(4, 1, figsize=(10, 8))
# Subtrama 1: Señal de Inphase component
wave_t[0].plot(tt, M)
wave_t[0].set_title('Señal Modulante')
wave_t[0].set_xlabel('time(sec)')
wave_t[0].set_ylabel('amplitude(volt0)')
wave_t[0].grid(True)

# Subtrama 2: Señal de Quadrature component
wave_t[1].plot(tt, s_FM)
wave_t[1].set_title('Señal a transmitir SIN ruido. Contiene las portadoras moduladas')
wave_t[1].set_xlabel('time(sec)')
wave_t[1].set_ylabel('amplitude(volt0)')
wave_t[1].grid(True)

# Subtrama 3: Señal de ruido
wave_t[2].plot(tt, noise)
wave_t[2].set_title('Ruido')
wave_t[2].set_xlabel('time(sec)')
wave_t[2].set_ylabel('amplitude(volt0)')
wave_t[2].grid(True)

# Subtrama 4: Señal final a transmitir
wave_t[3].plot(tt, s_FM_noise, color='red')
wave_t[3].set_title('Señal a transmitir CON ruido. Portadoras moduladas')
wave_t[3].set_xlabel('time(sec)')
wave_t[3].set_ylabel('amplitude(volt0)')
wave_t[3].grid(True)

# Ajustar diseño
plt.tight_layout()
plt.figure(1)

# Señal a transmitir luego de la modulacion
Tx_sig=s_FM_noise

# Señal final a transmitir Simbolos + Ruido
final_signal_with_noise = Tx_sig

# SOLO PARA TESTEAR
# Señal final a transmitir SOLO ruido
# final_signal_with_noise = y_noise

# Este valor es la mitad de 2475, es decir 1237.5 pero lo redondeo a 1238
# donde 2475 corresponde al intervalo de abscisas de la señal a transmitir
# Se considera la mitad ya que cuando se analiza el espectro los valores se espejan
# var_len_input = round(len(tt)/2)




# --/RECEPTOR/--

# --/Espectro FM Signal/--

f_fm = np.linspace(-fs/2, fs/2, len(final_signal_with_noise)) # Frequency Grid

f_fm_half = np.linspace(0, fs/2, round(len(final_signal_with_noise)/2))

M_f = fftshift(absolute(fft(M)))

C_f = fftshift(absolute(fft(C)))

S_FM_f = fftshift(absolute(fft(s_FM)))

S_FM_noise_f = fftshift(absolute(fft(s_FM_noise)))

# Señal recibida a la cual se le hizo transformada de fourier
# Y se reorganiza los componentes de frecuencia para que el 
# espectro esté centrado en cero.
signal_R_f = S_FM_noise_f

signal_R_f_half = S_FM_noise_f[round((fs/2)):]

#--/Calculo ancho de banda/--
beta = (kf*Am/(2*pi))/fm
BW = 2*fm*(beta+1)
BW_bound = np.zeros(len(f_fm))
for i in range(len(f_fm)):
    if(abs(f_fm[i])<(fc + BW/2) and abs(f_fm[i])>(fc - BW/2)):
        BW_bound[i] = 1
#--/Calculo ancho de banda/--
        
# --/Pruebas/--

valorFft = 32000

S_FM_f_fab = (absolute(fft(s_FM, valorFft)))

S_FM_f_fab_shift = fftshift(absolute(fft(s_FM, valorFft)))

S_FM_f_fab_noise2 = fftshift(absolute(fft(s_FM_noise, valorFft)))

tFab = np.arange(0,valorFft)

tFab2 = np.arange(-valorFft/2,valorFft/2)

# --/Pruebas/--

# --/Espectro FM Signal/--

# Valores de abscisas para la transformada de Fourier
# F = np.arange(0, var_len_input)

#Calculo de la transformada de Fourier de la señal recibida
# aux_noise = fourier.fft(final_signal_with_noise)
# fourier_signal_whit_noise = abs(aux_noise)

fig, fft_r = plt.subplots()
# Dividimos por el mayor valor de la señal recibida para nomralizar CREO
# fft_r.plot(f_fm, signal_R_f/signal_R_f.max())
fft_r.plot(f_fm_half, signal_R_f_half/signal_R_f_half.max())
# fft_r.set_xlabel('Frecuencia (Hz)[Señal con ruido]', fontsize='14')
# fft_r.set_ylabel('Amplitud FFT ', fontsize='14')
fft_r.set(xlabel = 'Frequency (Hz)', ylabel = 'Normalized |S_{FM}(f)|', xlim = (cota_inferior_frec,cota_superior_frec))
fft_r.set_title('Spectrum of FM Signal')
plt.title('Transformada de Fourier de la señal recibida')
plt.figure(2)


## Inicio de algoritmo LAD

# Arreglo con los valores de energía de la señal SUMANDO el ruido 
# (se considera la mitad del espectro ya que se espejan los valores)
# energy_of_signal_r = fourier_signal_whit_noise[:var_len_input]**2

# IMPORTANTE VER EL fftshift QUE SE COLOCO ABAJO
# energy_of_signal_r = fftshift(signal_R_f_half**2)
energy_of_signal_r = signal_R_f_half**2

# Creamos un arreglo de tuplas. En el primer valor cada tupla tendrá el valor de la energia de la señal, 
# en el segundo valor tendrá el indice en el espectro de frecuencia al que corresponde ese valor de energia.
# index = np.arange(0, len(energy_of_signal_r))
index = f_fm_half
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
plt.figure(3)

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
plt.figure(4)


fig, fft_r_lad_t = plt.subplots()
fft_r_lad_t.plot(f_fm_half, energy_of_signal_r)
fft_r_lad_t.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
fft_r_lad_t.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
# fft_r_lad_t.set_xlabel('Frecuencia (Hz)', fontsize='14')
# fft_r_lad_t.set_ylabel('Energia de la señal recibida', fontsize='14')
fft_r_lad_t.set(xlabel = 'Frecuencia (Hz)', ylabel = 'Energia de la señal recibida|', xlim = (cota_inferior_frec,cota_superior_frec))
plt.title('Energia de la señal recibida con umbrales del LAD')
plt.figure(5)


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
# print(clusters_list)

# BORRAR
# print(clusters_list_signal)

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
# print(signal_list)
      

fig, signals_by_lad_energy = plt.subplots()
# IMPORTANTE VER EL fftshift QUE SE COLOCO ABAJO
# signals_by_lad_energy.plot(f_fm_half, fftshift(energy_of_signal_r))
signals_by_lad_energy.plot(f_fm_half, energy_of_signal_r)
for i in range(0,len(signal_list)):
    signals_by_lad_energy.plot([x[1] for x in signal_list[i]], [y[0] for y in signal_list[i]])
signals_by_lad_energy.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
signals_by_lad_energy.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
# signals_by_lad_energy.set_xlabel('Frecuencia (Hz)', fontsize='14')
# signals_by_lad_energy.set_ylabel('Energia de la señal recibida', fontsize='14')
signals_by_lad_energy.set(xlabel = 'Frequency (Hz)', ylabel = 'Energia de la señal recibida', xlim = (cota_inferior_frec,cota_superior_frec))
plt.title('Señales identificadas mediante metodo LAD')
plt.figure(6)

# Señales detectadas con su valor debido a la Transformada de Fourier
signal_list_f = []

# Variable que va a almacenar cual es el mayor valor resultante de
# la Transformada de Fourier de todo el espectro
max_fft_value = 0

# Para completar el arraglo signal_list_f, tomo de cada una de las señales,
# el valor de energia le hago la raiz cuadrada (sqrt) y de esta forma
# pasan a ser los valor que se obtuvieron con la Transformada de Fourirer
for i in range(0,len(signal_list)):
   aux_signal = []
   
   for j in range(0,len(signal_list[i])):
      fft_value = np.sqrt(signal_list[i][j][0])
      aux_signal.append((fft_value,signal_list[i][j][1]))

      if((fft_value)>max_fft_value):
         max_fft_value = fft_value
   
   signal_list_f.append(aux_signal)


fig, signals_by_lad = plt.subplots()
# IMPORTANTE VER EL fftshift QUE SE COLOCO ABAJO
# signals_by_lad.plot(f_fm_half, fftshift(energy_of_signal_r))
signals_by_lad.plot(f_fm_half, signal_R_f_half/signal_R_f_half.max())
for i in range(0,len(signal_list_f)):
    signals_by_lad.plot([x[1] for x in signal_list_f[i]], [(y[0]/max_fft_value) for y in signal_list_f[i]])
# signals_by_lad.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
# signals_by_lad.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
# signals_by_lad.set_xlabel('Frecuencia (Hz)', fontsize='14')
# signals_by_lad.set_ylabel('Energia de la señal recibida', fontsize='14')
signals_by_lad.set(xlabel = 'Frequency (Hz)', ylabel = 'Normalized Transformada de Fourier de la señal recibida', xlim = (cota_inferior_frec,cota_superior_frec))
plt.title('Señales identificadas mediante metodo LAD')
plt.figure(7)


print("Cantidad de Señales detectadas: ")
print(len(signal_list))

# # Del arreglo de muestras ordenado de manera creciente segun su energía, solo consideramos
# # aquellos cuya energía es mayor al umbral inferior
# muestras_finales_aux = [tuple_item for tuple_item in sorted_array if tl <= tuple_item[0]]

# # Del arreglo anterior ahora solo considero aquellos cuya energía es mayor al umbral superior
# muestras_finales = [tuple_item for tuple_item in muestras_finales_aux if tu <= tuple_item[0]]

# # Extraigo el indice de cada una de las muestras cuya energía es mayor al umbral superior (por ende tambien mayor al umbral inferior)
# index_final_with_signal = [tuple_item[1] for tuple_item in muestras_finales]

# # Considerando la transformada de fourirer de la señal CON RUIDO
# fig, fft_r_lad = plt.subplots()
# fft_r_lad.plot(F, fourier_signal_whit_noise[:var_len_input])

# for index in range(0,len(index_final_with_signal)):
#   fft_r_lad.scatter(index_final_with_signal[index], 400, color='red', marker='o')
# fft_r_lad.set_xlabel('Frecuencia (Hz)', fontsize='14')
# fft_r_lad.set_ylabel('Amplitud FFT', fontsize='14')
# plt.title('Transformada de Fourier de la señal recibida mas metodo LAD')
# plt.figure(7)

# print('Cantidad de puntos rojos: ')
# print(len(index_final_with_signal))



plt.show()
