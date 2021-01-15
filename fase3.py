# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:23:04 2021

@author: isabe
"""
import tensorflow as tf 

from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import Dense 
import seaborn as sns 
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt  
import SwarmPackagePy 
from SwarmPackagePy import gsa 
from PySimpleGUI import PySimpleGUI as sg
from sklearn.linear_model import LogisticRegression 
# from SwarmPackagePy import testFunctions as tf 
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix 
from sklearn.neural_network import MLPClassifier

############# JANELA DE INPUT ######################
#Layout 
sg.theme('Reddit')
layout=[
    [sg.Text('Numero de agentes'),sg.Input(key='agentes',size=(20,1))],        
    [sg.Text('Dimensao'),sg.Input(key='dim',size=(20,1))], 
    [sg.Text('Numero de iteracoes'),sg.Input(key='it',size=(20,1))], 
    [sg.Text('Learning rate max'),sg.Input(key='lbb',size=(20,1))], 
    [sg.Text('Learning rate min'),sg.Input(key='ubb',size=(20,1))],
    [sg.Text('Numero de neuronios'),sg.Input(key='nn',size=(20,1))],
    [sg.Button('Ok')],
]
#Janela
janela=sg.Window('Tela',layout)
#Ler eventos
while True:
    eventos,valores=janela.read()
    if eventos==sg.WIN_CLOSED:
        break
    if eventos=='Ok':
        janela.close()
        print("Dataset Fashion-MNIST otimização do modelo")

############# JANELA DE INPUT ######################

# FASE 1 
###############
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print("train_images:", train_images.shape)
print("test_images:", test_images.shape)



# scale the values to a range of 0 to 1 of both data sets
train_images = train_images / 255.0
test_images = test_images / 255.0



# display the first 25 images from the training set and 
# display the class name below each image
plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5, i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
    

#############
# FASE 2 -- APLICAR RANDOM SEARCH GRID ??
##############

train_images = train_images.reshape(60000,28*28)
test_images = test_images.reshape(10000,28*28)

#CORRER COM 3 CAMADAS DE REDE
model = keras.Sequential([ 
    keras.layers.Flatten(input_shape=(28,28)), 
    keras.layers.Dense(128, activation=tf.nn.relu), 
    keras.layers.Dense(10, activation=tf.nn.softmax) 
]) 
model.summary()  
 
# Step 2 - Compile the model 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['acc']) 
 

model.fit(train_images, train_labels, epochs=5, validation_split=0.2) 
 
 

test_loss, test_acc = model.evaluate(test_images, test_labels) 
print("test loss:", test_loss * 100) 
print("test accuracy:", test_acc * 100) 
 
 
test_loss, test_acc = model.evaluate(test_images, test_labels) 
print("Using 3 layers - test loss:", test_loss * 100) 
print("Using 3 layers - test accuracy:", test_acc * 100) 
 


 

 
#Plot loss results for training data and testing data  
# plt.plot(model.history['loss'], 'blue') 
# plt.plot(model.history['val_loss'], 'orange') 
# plt.title('Model loss') 
# plt.ylabel('loss') 
# plt.xlabel('epoch') 
# plt.legend(['train', 'validate'], loc='upper left') 
# plt.show() 
 
# #Plot accuracy results for training data and testing data  
# plt.plot(model.history['acc'], 'green') 
# plt.plot(model.history['val_acc'], 'red') 
# plt.title('Model accuracy') 
# plt.ylabel('accuracy') 
# plt.xlabel('epoch') 
# plt.legend(['train', 'validate'], loc='upper left') 
# plt.show() 


 
predictions = model.predict(test_images) 





######################
# FASE 3- APLICAR OTIMIZADOR GSA
#Tentei trocar e experimentar o logistic aqui 
#######################
classifier = LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
classifier.fit(train_images,train_labels)
predicted = classifier.predict(test_images)
score_first = classifier.score(test_images, test_labels)
prediction=classifier.predict(train_images)
acc=confusion_matrix(test_images, prediction)
print("Accuracy: ",acc)
print("FIRST SCORE: ", score_first)

def log_class(hyperparamets):
    classifier = LogisticRegression(penalty='l2', dual=False, tol=(hyperparamets[0]), C=(hyperparamets[1]), fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    classifier.fit(train_images, train_labels)
    predicted = classifier.predict(test_images)
    score = classifier.score(test_images, test_labels)
    loss = (1.0-score)
    print("Loss: ",loss)
    print("\n")
    return loss

def f(x):
    
    valores.agentes = x.shape[0]
    print(valores.agentes)
    for i in [range(valores.agentes)]:
        valor=x[i]
          
    j = log_class(valor)
    print("-------aqui------")
    print(j)
    
    return np.array(j)


#################
def f1(x):
    
    nn = int(x[1])
          
    
    print("Learning rate: %.4f" %x[0])
    print("Neurons", nn)
    
    mlp=MLPClassifier(hidden_layer_sizes=nn,learning_rate_init=x[0])
    mlp.fit(train_images,train_labels)
    mlp.predict(test_images)
    acc=mlp.score(test_images,test_labels)
    print("ACC =%.4f" %acc)
    print("\n")
    
    return (1-acc)  
####################################
# In[14]:

# lbb = 0.01
# ubb= 2.0
# dimensions = ( 2 )
alg = SwarmPackagePy.gsa(n=20, lb = valores.lbb, ub = valores.ubb, function=f,  dimension=valores.dim, iterations=valores.it)

# lbb = 0.01
# ubb= 2.0
# dimensions = ( 2 )
alg1 = SwarmPackagePy.gsa(n=20, lb = valores.lbb, ub = valores.ubb, function=f1,  dimension=valores.dim, iterations=valores.it)
# In[15]:


um,dois= alg.get_Gbest()
print ("BEST RESULTS: ",alg.get_Gbest())

um = alg1.get_Gbest()
print ("BEST RESULTS FROM USING MLP: ",alg1.get_Gbest())

# In[16]:


# USING LOGISTIC
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', dual=False, tol=um, C=dois, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
classifier.fit(train_images,train_labels)
predicted = classifier.predict(test_images)
score_first = classifier.score(test_images,test_labels)
print("SECOND SCORE: ", score_first)







plt.figure(figsize=(15,10))
ax = sns.heatmap(confusion_matrix(test_labels,predicted),annot=True)
ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',
            xticklabels=(class_names),
            yticklabels=(class_names))