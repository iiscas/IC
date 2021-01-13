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
from sklearn.linear_model import LogisticRegression
# from SwarmPackagePy import testFunctions as tf
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


##############
# FASE 1 
###############
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# returns 4 numpy arrays: 2 training sets and 2 test sets
# images: 28x28 arrays, pixel values: 0 to 255
# labels: array of integers: 0 to 9 => class of clothings
# Training set: 60,000 images, Testing set: 10,000 images

# class names are not included, need to create them to plot the images  
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
    
# Step 1 - Build the architecture
# Model a simple 3-layer neural network
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

# #Step 3 - Train the model, by fitting it to the training data
# # 5 epochs, and split the training set into 80/20 for validation
model.fit(train_images, train_labels, epochs=5, validation_split=0.2)


# #Step 4 - Evaluate the model
# test_loss, test_acc = model_3.evaluate(test_images, test_labels)
# print("Model - 3 layers - test loss:", test_loss * 100)
# print("Model - 3 layers - test accuracy:", test_acc * 100)


test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Using 3 layers - test loss:", test_loss * 100)
print("Using 3 layers - test accuracy:", test_acc * 100)

# NN-3, 50 epochs 
history_NN3_50=model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

##Training again with diferent epochs
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Model - 3 layers - test loss:", test_loss * 100)
print("Model - 3 layers - test accuracy:", test_acc * 100)

#Plot loss results for training data and testing data 
plt.plot(history_NN3_50.history['loss'], 'blue')
plt.plot(history_NN3_50.history['val_loss'], 'orange')
plt.title('Model loss for the NN-3')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

#Plot accuracy results for training data and testing data 
plt.plot(history_NN3_50.history['acc'], 'green')
plt.plot(history_NN3_50.history['val_acc'], 'red')
plt.title('Model accuracy for the NN-3')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
# confidence of the model that the image corresponds to the label 
predictions = model.predict(test_images)
predictions.shape #(10000, 10)
predictions[0]

#############
# FASE 2 -- APLICAR RANDOM SEARCH GRID ??
##############

train_images = train_images.reshape(60000,28*28)
test_images = test_images.reshape(10000,28*28)



######################
# FASE 3- APLICAR OTIMIZADOR GSA
#Tentei trocar e experimentar o logistic aqui 
#######################
classifier = LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
classifier.fit(train_images,train_labels)
predicted = classifier.predict(test_images)
score_first = classifier.score(test_images, test_labels)
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
    
    n_particles = x.shape[0]
    print(n_particles)
    for i in [range(n_particles)]:
        valor=x[i]
          
    j = log_class(valor)
    print("-------aqui------")
    print(j)
    
    return np.array(j)

#PARA O MLP 
##################
# def f(x):
    
#     n_particles = float(x[1])
#     # j=[log_class(x[i]) for i in range(n_particles)]
          
#     # j = log_class(valor)
#     print("Learning rate: %.4f" %x[0])
#     print("Neurons", n_particles)
    
#     mlp=MLPClassifier(hidden_layer_sizes=n_particles,learning_rate_init=x[0])
#     mlp.fit(x_train,y_train)
#     # mlp.predict(x_test)
#     acc=mlp.score(x_test,y_test)
#     print("ACC =%.4f" %acc)
#     print("\n")
    
#     return (1-acc)  
#####################################
# In[14]:

#########
gsa(20, f, 0.01, 2.0, 2, 5)
lbb = 0.01
ubb= 2.0
dimensions = ( 2 )
alg = gsa(20, f, 0.01, 2.0, 2, 5)
alg.get_Gbest()
print(alg)

# In[15]:


um, dois = alg.get_Gbest()
print ("BEST RESULTS: ",alg.get_Gbest())


# In[16]:


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