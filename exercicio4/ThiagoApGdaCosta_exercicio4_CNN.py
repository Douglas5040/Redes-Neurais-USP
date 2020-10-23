# -*- coding: utf-8 -*-
"""
@Autor: Thiago Aparecido Gonçalves da Costa
@Disciplina: Redes Neurais

**********************EXERCÍCIO 4**********************

DATASET = MNIST
TIPO DE REDE NEURAL: CNN

VALIDAÇÃO: CLASSIFICAÇÃO DE NÚMEROS ESCRITOS A MÃO
"""

#***************************BIBLIOTECAS*******************************
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
#from keras import backend as K
from tensorflow.compat.v1.keras import backend as K

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import numpy as np
from PIL import Image
import sys
import cv2

print("----> Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#**********PRÉ-PROCESSAMENTO DOS DADOS******
batch_size = 512
num_classes = 10
epocas = 10
memory_limit=2048 #pelo motivo da GPU ser usada no comoutador para GUI e outros

#************DIMENSÕES DA IMAGEM*************
linhas, colunas = 28, 28

#**********CARREGANDO A BASE DE DADOS*********
dados = mnist.load_data()

#******DIVISÃO EM TREINAMENTO E TESTE*********
(x_treinamento, y_treinamento), (x_teste, y_teste) = dados

#***REDIMENSIONAMENTO DAS BASES DE TREINAMENTO E TESTE******

#***REDIMENSIONAMENTO DAS BASES DE TREINAMENTO E TESTE*********
x_treinamento = x_treinamento.reshape(x_treinamento.shape[0], linhas, colunas, 1)
x_teste = x_teste.reshape(x_teste.shape[0], linhas, colunas, 1)
input_shape = (linhas, colunas, 1)

#*************NORMALIZAÇÃO*************
x_treinamento = x_treinamento.astype('float32')
x_teste = x_teste.astype('float32')
x_treinamento /= 255
x_teste /= 255

print('CONJUNTO DE DADOS:', x_treinamento.shape)
print('TREINAMENTO: ', x_treinamento.shape[0])
print('TESTE: ', x_teste.shape[0])

#*******************************GPU*******************************

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

#Criando sessão para que o tensorflow + cuda rode a CNN na GPU
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
K.set_session(session)
#run_opts = tf.distribute.RunOptions(experimental_bucketizing_dynamic_shape = True)

# config = ConfigProto()
# config.gpu_options.visible_device_list = "0,1"
# with tf.Session(config) as sess:
#or K.set_session(tf.Session(config))
#******************************************************************

y_treinamento = keras.utils.to_categorical(y_treinamento, num_classes)
y_teste = keras.utils.to_categorical(y_teste, num_classes)

#*******************************CNN********************************
model = Sequential()

#*************************CAMADAS CONVOLUCIONAIS*******************
model.add(Conv2D(128, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(128, (5, 5), activation='relu'))
#*************************MaxPooling2D*******************
#Extração de características: Polling
model.add(MaxPooling2D(pool_size=(2, 2)))
#*************************CAMADAS CONVOLUCIONAIS*******************
model.add(Conv2D(256, (2, 2), activation='relu'))
model.add(Conv2D(256, (2, 2), activation='relu'))
#*************************MaxPooling2D*******************
model.add(MaxPooling2D(pool_size=(2, 2)))
#*************************CAMADAS CONVOLUCIONAIS*******************
model.add(Conv2D(512, (2, 2), activation='relu'))
model.add(Conv2D(512, (2, 2), activation='relu'))
#*************************MaxPooling2D*******************
#Extração de características: Polling
model.add(MaxPooling2D(pool_size=(2, 2)))
#*************************Dropout*******************
model.add(Dropout(0.5))
#*************************Flatten*******************
model.add(Flatten())

#****************CAMADAS TOTALMENTE CONECTADAS********************
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))

#***********************CAMADA DE SAÍDA****************************
model.add(Dense(num_classes, activation='softmax'))

#***********************TREINAMENTO*********************************
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_treinamento, y_treinamento,
          batch_size=batch_size,
          epochs=epocas,
          verbose=1,
          validation_data=(x_teste, y_teste))

#***********************TESTE*********************************
score = model.evaluate(x_teste, y_teste, verbose=0)

print('% DE PERDA:', score[0]*100)
print('% DE ACURÁCIA:', score[1]*100)

#*************LEITURA DAS FOTOS*************
#FOTOS SEM RUÍDO

digitos_teste = [4,5,8,9]

for i in digitos_teste:

    try:
        img = cv2.imread('fotos_digitos/'+str(i)+'.jpeg',0)
        kernel = np.ones((5,5), np.uint8)
        #APLICAÇÃO DE EROSÃO E DILATAÇÃO NA IMAGEM PARA RETIRAR RUÍDOS E REALÇAR AS CARACTERÍSTICAS PRINCIPAIS
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        #INVERSÃO DAS CORES, VISTO QUE O NÚMERO DEVE ESTAR COM COR CLARA
        img = cv2.bitwise_not(opening)
        digito = Image.fromarray(img)

    except IOError:
        print("Unable to load image")
        sys.exit(1)

    #**********VISUALIZAÇÕES DAS MODIFICAÇÕES FEITAS PELO OPENCV************
    digito.show()

    #**********NORMALIZAÇÃO DO TESTE************
    digito = digito.resize((28, 28), Image.ANTIALIAS)

    digito = np.array(digito)

    digito = digito.reshape(1, linhas, colunas, 1)

    digito = digito.astype('float32')
    digito /= 255

    #**********INFERÊNCIA DO TESTE************
    predictions_single = model.predict(digito)

    resultado = np.argmax(predictions_single[0])

    print('SAÍDA DESEJADA', i)
    print('SAÍDA', resultado)
