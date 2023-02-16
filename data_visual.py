'''
Created on 14-Jun-2021

@author: somsh
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from Model import config

# d=dtale.show(df)
# print(d)
# d.open_browser()
choice="Y"
while choice=="Y":
    Axes=input("Enter axis: ").upper()
    Class=input("Enter Class: ").lower()
    # Class="A"
    pathname=config.BASE_PATH+"/"+Axes+"/"+Class+".csv"
    df=pd.read_csv(pathname,header=None)
    index=int(input("Enter Signal index: "))
    A=np.array(df.iloc[index])
    N=len(A)
    plt.figure()
    plt.plot(A)
    A_F=fft(A)
    realA_F=[i.real for i in A_F]
    imagA_F=[i.imag for i in A_F]
    plt.figure()
    plt.plot(realA_F,imagA_F,color="red")
    plt.figure()
    plt.polar(np.angle(A_F),np.abs(A_F))
    A_F=A_F[:len(A_F)//2]
    i=np.argmax(np.abs(A_F))
    print(i)
    choice=input("continue? Y/N: ").upper()
    plt.figure()
    plt.plot(np.abs(A_F))
    # plt.figure()
    # plt.plot(np.angle(A_F))
    plt.show()


