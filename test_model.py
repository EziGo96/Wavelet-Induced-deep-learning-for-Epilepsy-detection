'''
Created on 07-Apr-2020

@author: somsh
'''
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from Model import config
import pandas as pd
from sklearn.model_selection import train_test_split
from Wavelet_Kernels.MexicanHat import Mexh_filter
from Wavelet_Kernels.Morlet import Morlet_filter
from Wavelet_Kernels.Laplace import Laplace_filter
from Wavelet_Kernels.Gauss import Gaussian_filter
from Wavelet_Kernels.Daubechies import *
from Wavelet_Kernels.Symlet import *
from tensorflow.keras.models import model_from_json
BS = 20
Classes=["focal","nonfocal"]
X={}
for i in Classes:
    X[i]=pd.read_csv(config.BASE_PATH+"/"+config.ORIG_INPUT_DATASET_X+"/"+i+".csv",header=None).values
Y={}
for i in Classes:
    Y[i]=[Classes.index(i) for _ in X[i]]
    
X_train={}
X_test={}
Y_train={}
Y_test={}
for j,k in zip(X.keys(),Y.keys()):
    X_train[j], X_test[j], Y_train[k], Y_test[k] = train_test_split(X[j], Y[k], train_size=config.TRAIN_SPLIT, random_state=32)
    
arrays=[X_test[_] for _ in X_test.keys()]
X_test_DS=np.concatenate(arrays, axis=0).astype('float32')
X_test_DS=np.reshape(X_test_DS,(X_test_DS.shape[0],X_test_DS.shape[1],1))
arrays=[Y_test[_] for _ in Y_test.keys()]
Y_test_DS=np.concatenate(arrays, axis=0)
Y_test_DS=np.reshape(Y_test_DS,(Y_test_DS.shape[0],1))

model=load_model('model.hp5',custom_objects={'Mexh_filter':Mexh_filter,'Morlet_filter':Morlet_filter,'Laplace_filter':Laplace_filter,'Gaussian_filter': Gaussian_filter, 'tf': tf})
json_file = open('architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json, custom_objects={'Mexh_filter':Mexh_filter,'Morlet_filter':Morlet_filter,'Laplace_filter':Laplace_filter,'Gaussian_filter': Gaussian_filter, 'tf': tf})
model1.load_weights("model_wts.hdf5")
# Testing
totalTest = len(list(X_test_DS))
# initialize the testing generator
# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
predIdxs = model.predict(X_test_DS,batch_size = BS)
predIdxs1 = model1.predict(X_test_DS,batch_size = BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
predIdxs1 = np.argmax(predIdxs1, axis=1)

# show a confusion matrix and formatted classification report
print(confusion_matrix(Y_test_DS, predIdxs))
print(classification_report(Y_test_DS, predIdxs))
print("\n")
print(confusion_matrix(Y_test_DS, predIdxs1))
print(classification_report(Y_test_DS, predIdxs1))