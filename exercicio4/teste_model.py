
import tensorflow as tf
from keras.models import model_from_json
from PIL import Image
import numpy as np
import time
import sys
import cv2
import os


#Dimensões da imagem
linhas, colunas = 28, 28
#Estamos colocando esse limite de memória pelo motivo da GPU ser usada no computador para GUI e outros
memory_limit=512 

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

#Inferência dos dados
digitos_teste = [0,1,2,3,4,5,6,8,9]

#CARREGANDO MODELO 
# load json and create model
json_file = open('modelo/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelo/model.h5")
print("Loaded model from disk")

model = loaded_model

# evaluate loaded model on test data
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
# score = model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

for i in digitos_teste:

    try:
        img = cv2.imread('fotos_numeros/'+str(i)+'.jpg',0)
        #img2 = mpimg.imread('fotos_numeros/'+str(i)+'.jpg',0)
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

    digito = np.array(digito)

    digito = digito.reshape(1, linhas, colunas, 1)

    digito = digito.astype('float32')
    digito /= 255

    #Inferência do teste
    predictions_single = model.predict(digito)

    resultado = np.argmax(predictions_single[0])

    print('SAÍDA DESEJADA', i)
    print('SAÍDA', resultado)
