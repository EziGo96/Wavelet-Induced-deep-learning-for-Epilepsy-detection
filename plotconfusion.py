'''
Created on 23-Jul-2021

@author: somsh
'''
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
from pandas import DataFrame as df
import numpy as np
Classes=["A","B","C","D","E"]
array = np.array([[19 , 1 , 0 , 0 , 0],
 [ 1 ,19  ,0  ,0 , 0],
 [ 1 , 0 ,18 , 1 , 0],
 [ 0 , 0 , 1, 19 , 0],
 [ 0 , 0 , 0 , 0 ,20]])
#get pandas dataframe
df_cm = df(array, index=Classes, columns=Classes)
#colormap: see this and choose your more dear
pretty_plot_confusion_matrix(df_cm)