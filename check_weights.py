'''
Created on 22-Jun-2021

@author: somsh
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from Model.Resnet import Mexh_filter
model=load_model('model.hp5',custom_objects={'Mexh_filter': Mexh_filter, 'tf': tf})
weights_list = model.get_weights()
print (np.array(weights_list,dtype='object').shape)