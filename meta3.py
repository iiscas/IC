from __future__ import print_function

# Keras Models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import keras
# Aditional Libs
import numpy as np
import os
import pytest
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# from SwarmPackagePy import gsa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

from keras import Sequential
from keras.layers import Dense

import SwarmPackagePy
from SwarmPackagePy import animation
from SwarmPackagePy import gsa

from SwarmPackagePy import testFunctions as tf

import regex

# x_train, y_train, x_valid, y_valid, x_test ,y_test = [], [], [], [], [], []
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3)

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

x_train=x_train/255.0
x_test=x_test/255.0

x_train = x_train.reshape(len(x_train),28*28)
x_test = x_test.reshape(10000,28*28)




print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')




###########################
# 4: Treinar rede
###########################

classifier = LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=5, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
classifier.fit(x_train,y_train)
predicted = classifier.predict(x_test)
score_first = classifier.score(x_test,y_test)
print("FIRST SCORE: ", score_first)

#def f():
clf = 
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
acc=confusion_matrix(y_test, prediction)
print (acc)


def f(x):
    
    n_particles = float(x[1])
    # j=[log_class(x[i]) for i in range(n_particles)]
          
    # j = log_class(valor)
    print("Learning rate: %.4f" %x[0])
    print("Neurons", n_particles)
    
    mlp=MLPClassifier(hidden_layer_sizes=n_particles,learning_rate_init=x[0])
    mlp.fit(x_train,y_train)
    # mlp.predict(x_test)
    acc=mlp.score(x_test,y_test)
    print("ACC =%.4f" %acc)
    print("\n")
    
    return (1-acc)    


lbb = 0.01
ubb= 2.0
dimensions = ( 2 )
alg = gsa(n=20,function=f ,lb = lbb, ub = ubb,  dimension=dimensions, iteration=5,G0=3)
animation(alg.get_agents(), tf.easom_function, -10, 10)
print ("BEST RESULTS: ",alg.get_Gbest())
um, dois = alg.get_Gbest()
print ("BEST RESULTS: ",alg.get_Gbest())