import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

INPUT_SHAPE = 784
NUM_CATEGORIES = 10

LABEL_DICT = {
 0: "T-shirt/top",
 1: "Trouser",
 2: "Pullover",
 3: "Dress",
 4: "Coat",
 5: "Sandal",
 6: "Shirt",
 7: "Sneaker",
 8: "Bag",
 9: "Ankle boot"
}

# LOAD THE RAW DATA
train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

print(train.shape)
print(test.shape)



X = train.iloc[:,1:] 
Y = train.iloc[:,0] 


x_train , x_test , y_train , y_test = train_test_split(X, Y , test_size=0.1)

class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X.loc[i].values.reshape((28,28))) 
    label_index = int(Y[i]) 
    plt.title(class_names[label_index])
plt.show()


x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)


x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255


y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(y_train[0])
print(y_test[0])



classifier = LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
classifier.fit(x_train, y_train)
predicted = classifier.predict(x_test)
score_first = classifier.score(x_test, y_test)
print("FIRST SCORE: ", score_first)

def log_class(hyperparamets):
    classifier = LogisticRegression(penalty='l2', dual=False, tol=(hyperparamets[0]), C=(hyperparamets[1]), fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    score = classifier.score(x_test, y_test)
    loss = (1.0-score)
    return loss


# def accuracy(confusion_matrix):
#     diagonal_sum = confusion_matrix.trace()
#     sum_of_all_elements = confusion_matrix.sum()

#     return diagonal_sum / sum_of_all_elements


def f(x):
    
    print("valor de x= ", x)
    n_particles = x.shape[0]

    for i in [range(n_particles)]:
        valor=x[i]
          
    j = log_class(valor)
    return np.array(j)

lbb = 0.01
ubb= 2.0
dimensions = ( 2 )
alg = SwarmPackagePy.gsa(n=10, lb = lbb, ub = ubb, function=f,  dimension=dimensions, iteration=10)


um, dois = alg.get_Gbest()
print ("BEST RESULTS: ",alg.get_Gbest())