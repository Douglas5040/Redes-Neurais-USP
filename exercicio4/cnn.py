'''
Universidade de São Paulo (USP) – Pós-Graduação ICMC – São Carlos

Disciplina: SCC5809 – Redes Neurais Artificiais

Profa. Dra. Roseli Aparecida Francelin Romero

Equipe:
Dheniffer Caroline Araújo Pessoa - Nº USP - 12116252
Douglas Queiroz Galucio Batista - Nº USP - 12114819
Laleska Aparecida Ferreira Mesquita Nº USP - 12116738'''

#Importação das bibliotecas necessárias
import tensorflow as tf
import keras
from keras.optimizers import Adam
#Do Keras estamos importando o 'backend' e estamos dando o apelido de 'k'
from tensorflow.compat.v1.keras import backend as K

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import numpy as np
from PIL import Image
import sys
import cv2
import time
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("----> Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Aqui estamos fazendo o pré-processamento de dados
batch_size = 512
num_classes = 10
epocas = 1000
memory_limit=2048  #Estamos colocando esse limite de memória pelo motivo da GPU ser usada no computador para GUI e outros

#Dimensões da imagem
linhas, colunas = 28, 28

#Aqui estamos carregando a base de dados
dados = keras.datasets.mnist.load_data()

#Aqui estamos fazendo a divisão entre treinamento e teste
(x_treinamento, y_treinamento), (x_teste, y_teste) = dados

#Aqui estamos declarando o redimensionamento das bases de treinamento e teste
x_treinamento = x_treinamento.reshape(x_treinamento.shape[0], linhas, colunas, 1)
x_teste = x_teste.reshape(x_teste.shape[0], linhas, colunas, 1)
input_shape = (linhas, colunas, 1)

#Normalização
x_treinamento = x_treinamento.astype('float32')
x_teste = x_teste.astype('float32')
x_treinamento /= 255
x_teste /= 255

print('CONJUNTO DE DADOS:', x_treinamento.shape)
print('TREINAMENTO: ', x_treinamento.shape[0])
print('TESTE: ', x_teste.shape[0])

#GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Nesse método estamos restringindo o Tensorflow para alocar apenas 1 GB de memória na primeira GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # É importante saber que os dispositivos virtuais devem ser configurados antes que as GPUs sejam inicializadas
    print(e)

#Aqui estamos criando uma sessão para que o tensorflow + cuda rode a CNN na GPU
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
K.set_session(session)

y_treinamento = keras.utils.to_categorical(y_treinamento, num_classes)
y_teste = keras.utils.to_categorical(y_teste, num_classes)

#CNN
model = keras.models.Sequential()

#Camadas Convolucionais
model.add(keras.layers.Conv2D(128, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(keras.layers.Conv2D(128, (5, 5), activation='relu'))

#MaxPooling2D
#Extração de características: Polling
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

#Camadas Convolucionais
model.add(keras.layers.Conv2D(256, (2, 2), activation='relu'))
model.add(keras.layers.Conv2D(256, (2, 2), activation='relu'))

#MaxPooling2D
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

#Camadas Convolucionais
model.add(keras.layers.Conv2D(512, (2, 2), activation='relu'))
model.add(keras.layers.Conv2D(512, (2, 2), activation='relu'))

#MaxPooling2D
#Aqui estamos fazendo a extração de características: Polling
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

#Dropout
model.add(keras.layers.Dropout(0.5))
#keras.layers.Flatten
model.add(keras.layers.Flatten())

#Aqui estamos adicionando camadas totalmente conectadas
model.add(keras.layers.Dense(2048, activation='relu'))
model.add(keras.layers.Dense(2048, activation='relu'))
model.add(keras.layers.Dropout(0.5))

#Camada de saída
model.add(keras.layers.Dense(num_classes, activation='softmax'))

#Aqui estamos fazendo o treinamento
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_treinamento, y_treinamento,
          batch_size=batch_size,
          epochs=epocas,
          verbose=1,
          validation_data=(x_teste, y_teste))

#Aqui estamos fazendo o teste
score = model.evaluate(x_teste, y_teste, verbose=0)

print('% DE PERDA:', score[0]*100)
print('% DE ACURÁCIA:', score[1]*100)

#Aqui estamos fazendo a leitura das fotos
#As fotos estão sem ruídos

digitos_teste = [0,1,2,3,4,5,6,8,9]

#SALVANDO MODELO 
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelo/model.h5")
print("Saved model to disk")

for i in digitos_teste:

    try:
        img = cv2.imread('fotos_numeros/'+str(i)+'.jpg',0)
        img2 = mpimg.imread('fotos_numeros/'+str(i)+'.jpg',0)
        kernel = np.ones((5,5), np.uint8)
        # Aqui estamos fazendo a aplicação de erosão e dilatação na imagem, para assim retirar os ruídos e realçar as características principais
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        #Aqui estamos invertendo as cores, pois o número deve estar com a cor clara
        img = cv2.bitwise_not(opening)
        digito = Image.fromarray(img)

    except IOError:
        print("Unable to load image")
        sys.exit(1)

    #Através disso é possível visualizar as modificações feitas pelo OpenCV
    os.environ['DISPLAY'] = ':1'
    #digito.show()
    #time.sleep(2)

    #Normalização do teste
    digito = digito.resize((28, 28), Image.ANTIALIAS)

    digito.show()
    
    digito = np.array(digito)

    digito = digito.reshape(1, linhas, colunas, 1)
    

    digito = digito.astype('float32')
    digito /= 255

    #Inferência do teste
    predictions_single = model.predict(digito)

    resultado = np.argmax(predictions_single[0])

    print('SAÍDA DESEJADA', i)
    print('SAÍDA', resultado)
