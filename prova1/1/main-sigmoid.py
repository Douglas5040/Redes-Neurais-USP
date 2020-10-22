
import numpy as np
import sklearn as sk
import sklearn.utils as sk_utils
from sklearn.model_selection import train_test_split
from random import uniform
import pandas as pd
import decimal
import matplotlib.pyplot as plt

#**********************FUNÇÃO DE ATIVAÇÃO***************************
def getFnet(net):
    return (1 / (1 + np.exp(-net)))
    
#**********************DERIVADA DA FUNÇÃO DE ATIVAÇÃO***************
def getDerFnet(fnet):
    return (np.multiply(fnet, (1 - fnet)))

#**********************MODELO INICIAL DA ARQUITETURA****************
class MLP_Arquitetura:
    
    def __init__(self, entrada, oculta, saida):
        
        self.entrada = entrada
        self.oculta = oculta
        self.saida = saida
        self.hidden_model = self.initHiddenModel()
        self.output_model = self.initOutputModel()
        
    #**********************INICIAR O MODELO DA CAMADA OCULTA********
    def initHiddenModel(self):
         
        #************PESOS E THETAS PARA A CAMADA OCULTA************
        vetor_oculta = np.array([])
        cont = 0
        while(cont < (self.oculta * (self.entrada+1))):
            vetor_oculta = np.append(vetor_oculta,uniform(-1,1))
            cont+=1
         
        aux = vetor_oculta.reshape(self.oculta, (self.entrada+1))
        return np.asmatrix(aux)
        
    #**********************INICIAR O MODELO DA CAMADA DE SAÍDA******
    def initOutputModel(self):
        
        #************PESOS E THETAS PARA A CAMADA DE SAÍDA**********
        vetor_saida = np.array([])
        cont = 0
        while(cont < (self.saida * (self.oculta+1))):
            vetor_saida = np.append(vetor_saida,uniform(-1,1))
            cont+=1
         
        aux = vetor_saida.reshape(self.saida, (self.oculta+1))
        return np.asmatrix(aux)

#**********************FNET = ONDE GUARDO OS RESULTADOS**********************
class Fnet():

    net_hidden = []
    fnet_hidden = []
    net_output = []
    fnet_output = []

##########################################################################################

# NET = SOMATÓRIA DOS PESOS PELA ENTRADA
def MLP_Forward(model, teste, fnet):
	
	#**********VETOR DE ENTRADAS**********
    teste = np.squeeze(np.asarray(teste))
    teste = np.append(teste,1)
	
	#*****VETOR DE ENTRADAS TRANSPOSTO*****
    hidden  = np.asmatrix(model.hidden_model)
    
    teste = np.asmatrix(teste)
    teste = np.transpose(teste)
    
    #********** SAÍDA DA CAMADA OCULTA**********
    net_hidden = hidden*teste
    
    fnet.net_hidden = net_hidden

    fnet_hidden = getFnet(net_hidden)
    fnet.fnet_hidden = fnet_hidden
    
    #*******************************************
    output  = model.output_model
    output = np.asmatrix(output)
    
    fnet_hidden = np.squeeze(np.asarray(fnet_hidden))
    fnet_hidden = np.append(fnet_hidden,1)
    fnet_hidden = np.asmatrix(fnet_hidden)      
    fnet_hidden = np.transpose(fnet_hidden)
    
    #********** SAÍDA DA CAMADA DE SAÍDA**********
    net_output = output*fnet_hidden
    
    fnet.net_output = net_output      
    

    fnet_output = getFnet(net_output)
    
    fnet.fnet_output = fnet_output
    #*******************************************

    return fnet

def MLP_Backpropagation(model, dataset, eta, limiar, epocas):

        limiar_backpropagation = 2 * limiar
        cont_epocas = 0
        dados_saida1 = []
        dados_saida2 = []
        resultados = Fnet()
		
        while(limiar_backpropagation > limiar and cont_epocas <= epocas):
            
            limiar_backpropagation = 0
            dataset=np.asmatrix(dataset)

            for linha in range(dataset.A.shape[0]):
                
                entradas_desejadas = dataset[linha,0:model.entrada]
                saidas_desejadas =   dataset[linha,model.entrada:dataset.A.shape[1]]
                
                #*********************FORWARD**********************
                resultados = MLP_Forward(model , entradas_desejadas,resultados)

                
                # print("Saida camada intermediaria: ", dados_saida1)
                
                #*********************ERROS************************
                error = saidas_desejadas - np.transpose(resultados.fnet_output)
                
                limiar_backpropagation += np.sum(np.power(error,2))
                
                #*********************DELTA DA SAÍDA************************
				    # delta_o = (Yp - Op) * f_o_p'(net_o_p)
                delta_saida = np.multiply(error , np.transpose(getDerFnet(resultados.fnet_output)))
                
                #***********************************************************
                
                #*********************DELTA DA OCULTA***********************
                # delta_hidden = f_h_p'(net_h_p) * sum =(delta_o * w_o_kj)
                w_o=np.asmatrix(model.output_model[:,0:model.oculta])
                
                delta_oculta = np.multiply(np.transpose(getDerFnet(resultados.fnet_hidden)),(delta_saida*w_o))
                
				    #*********************TREINAMENTO***************************
                fnet1 = np.squeeze(np.asarray(resultados.fnet_hidden))
                fnet1 = np.append(fnet1,1)
                fnet1 = np.asmatrix(fnet1)

                print("Saida camada intermediaria: ", fnet1[0,0])
                dados_saida1.append(fnet1[0,0])
                print("Saida camada intermediaria: ", fnet1[0,1])
                dados_saida2.append(fnet1[0,1])

                print(fnet1[0,1])
                # Treinamento da camada de saída
                # w(t+1) = w(t) + eta * dE2_o * i_pj
                
                #************************momentum * modelo***************************
                aux_eta = eta*(np.transpose(delta_saida)*fnet1)
                
                #*****MODELO OUTPU ANTIGO*****
                modelo_output_antigo  = model.output_model
                
                #*****MODELO OUTPUT NOVO*****
                modelo_output_novo  = model.output_model+aux_eta
                
                #*****INCLUSÃO DO TERMO MOMENTUM E ATUALIZAÇÃO DOS PESOS*****
                model.output_model = model.output_model + (modelo_output_novo - modelo_output_antigo) * 0.5
                
                x1=np.squeeze(np.asarray(entradas_desejadas))
                x1=np.append(x1,1)
                x1=np.asmatrix(x1) 
                
                #************************momentum * modelo***************************
                
                aux_eta = eta*(np.transpose(delta_oculta)*x1)
                
                model.hidden_model = model.hidden_model+aux_eta
                
                #*****MODELO OUTPU ANTIGO*****
                modelo_hidden_antigo  =  model.hidden_model
                
                #*****MODELO OUTPUT NOVO*****
                modelo_hidden_novo  = model.hidden_model+aux_eta
                
                #*****INCLUSÃO DO TERMO MOMENTUM E ATUALIZAÇÃO DOS PESOS*****
                model.hidden_model = model.hidden_model + (modelo_hidden_novo - modelo_hidden_antigo) * 0.5
                
                limiar_backpropagation = limiar_backpropagation / dataset.A.shape[0]
                   
            #print('Erro Médio Quadrático = ',limiar_backpropagation)
            cont_epocas +=1

        print('--------------TERMINOU--------------')
        print('Quantidade de Épocas = ',cont_epocas)
        #print(dados_saida1)

        plt.plot(dados_saida1, label='MLP Treinamento')
        plt.plot(dados_saida2, label='MLP')

        plt.xlabel('quantidade dados')
        plt.ylabel('Acurácia (%)')
        plt.title("Dados das duas saidas")
        plt.tight_layout()
        plt.show()
        
        if(cont_epocas < epocas):
            print('--------------TREINADO--------------')
        
        return(model, resultados, cont_epocas)

##########################################################################################

#QUANTIDADE DE NEURÔNIOS NAS CAMADAS
qtd_entradas = 7
qtd_saidas = 3
v_ocultas = [2]

#TAXA DE APRENDIZADO
eta = 0.5

#LIMITE MÁXIMO DE ÉPOCAS
epocas = 10

#LIMITE MÁXIMO DE ÉPOCAS
porc_teste = 0.2

#LIMIAR 
limiar = 0.00001
    
#****************************************LEITURA DA BASE DE DADOS**********************************
dataset = pd.read_csv('seeds.csv')


labels = dataset['S1'].values
labels = set(labels)

#*********************************************PRÉ-PROCESSAMENTO************************************

dataset_bin = pd.DataFrame(columns= ['E1','E2','E3','E4','E5','E6','E7','S1','S2','S3'])

for x in labels:
    
    d = dataset[dataset['S1'] == x]
    
    if(x == 1):
        d.loc[:,'S1'] = 0
        d.loc[:,'S2'] = 0
        d.loc[:,'S3'] = 1
        
    elif (x == 2):
        d.loc[:,'S1'] = 0
        d.loc[:,'S2'] = 1
        d.loc[:,'S3'] = 0
        
    else:
        d.loc[:,'S1'] = 1
        d.loc[:,'S2'] = 0
        d.loc[:,'S3'] = 0
    
    dataset_bin = pd.concat([dataset_bin, d])


#********************************NORMALIZAÇÃO ENTRE 0 E 1 DA BASE DE DADOS***************************
    
normalizer = sk.preprocessing.MinMaxScaler(feature_range = (0,1))
dataset_normalizado = normalizer.fit_transform(dataset_bin)
dataset_normalizado = pd.DataFrame(dataset_normalizado, columns= ['E1','E2','E3','E4','E5','E6','E7','S1','S2','S3'])

for o in v_ocultas:
    
    qtd_ocultas = o
    
    #**************************************************EMBARALHANDO**************************************
       
    dataset_teste = pd.DataFrame(columns= ['E1','E2','E3','E4','E5','E6','E7','S1','S2','S3'])
    dataset_treinamento = pd.DataFrame(columns= ['E1','E2','E3','E4','E5','E6','E7','S1','S2','S3'])
    
    dataset_normalizado = sk_utils.shuffle(dataset_normalizado)
    
    #********************************SELECIONANDO A PARCELA PARA TREINAMENTO E TESTE**********************
    
    saidas = ['S1','S2','S3']
    
    for x in saidas:
    
        d = dataset_normalizado[dataset_normalizado[x] == 1]    
        
        treinamento, teste = train_test_split(d, test_size=porc_teste)
        
        dataset_teste = dataset_teste.append(teste)
        
        dataset_treinamento = dataset_treinamento.append(treinamento)
    
    
    mlp = MLP_Arquitetura(qtd_entradas,qtd_ocultas,qtd_saidas)
    
    dataset_treinamento=np.asmatrix(dataset_treinamento)
    dataset_teste=np.asmatrix(dataset_teste)
    
    modelo, resultados, cont_epocas = MLP_Backpropagation(mlp, dataset_treinamento, eta, limiar , epocas)
    
    #*******************************************ACURÁCIA**********************************************
    somatoria = 0
    for linha in range(dataset_teste.A.shape[0]):
        aux_sd = dataset_teste[linha,mlp.entrada:dataset_teste.A.shape[1]]
        aux_sd = np.squeeze(np.asarray(aux_sd))
        
        aux_saida = MLP_Forward(modelo,dataset_teste[linha,0:mlp.entrada],resultados)
        aux_saida = np.squeeze(np.asarray(aux_saida.fnet_output))       
        aux_saida =  np.around(aux_saida)
        
        #COMPARAÇÃO SE O VETOR DE SAÍDA É IGUAL AO DESEJADO 
        sum_equal = 0
        for i in range(len(aux_sd) - 1):
            if(aux_sd[i] != aux_saida[i]):
                sum_equal = 1
                break
        #CASO O VETOR DE SAÍDA SEJA IGUAL, LOGO SOMO 1       
        if(sum_equal == 0):
            somatoria +=1
            
        #print('Saida Desejada: '+str(aux_sd)+' Saida Obtida: '+str(aux_saida))
    
    # ACURACIA É A RAZÃO ENTRE A "SOMATORIA" E A QUANTIDADE DE EXEMPLOS DE TESTES
    acuracia = (somatoria / dataset_teste.A.shape[0]) * 100
    
    d = decimal.Decimal(acuracia)
    acuracia = round(d,2)    
    
    #***************************************************************************************************
    
    print('Qtd de neurônios na camada de entrada: '+str(qtd_entradas))
    print('Qtd de neurônios na camada oculta: '+str(qtd_ocultas))
    print('Qtd de neurônios na camada de saída: '+str(qtd_saidas))
    print('Parcela de teste: '+str(porc_teste))
    print('Taxa de aprendizado: '+str(eta))
    print('Acurácia: '+str(acuracia))
    print('Quantidade de epocas: '+str(cont_epocas))
