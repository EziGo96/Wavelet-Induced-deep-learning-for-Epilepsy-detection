'''
Created on 14-Jun-2021

@author: somsh
'''
import argparse
from Model import config
# from Model.Resnet import ResNet
from Model.AlexNet import Alexnet
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import save_model
import matplotlib
from matplotlib import pyplot as plt
from pandas import DataFrame as df
from scipy.io import savemat

Classes=["focal","nonfocal"]
X={}
for i in Classes:
	X[i]=pd.read_csv(config.BASE_PATH+"/"+config.ORIG_INPUT_DATASET_X+"/"+i+".csv",header=None).values
Y={}
for i in Classes:
	l=[]
	for _ in X[i]:
		l.append(Classes.index(i))
	for _ in range(len(l)):
		shape=(len(Classes))
		one_hot=np.zeros(shape)
		one_hot[l[_]]=1
		l[_]=one_hot
	Y[i]=np.array(l)
	
X_train={}
X_test={}
Y_train={}
Y_test={}
X_val={}
Y_val={}

matplotlib.use("Agg")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

NUM_EPOCHS = 100
INIT_LR = 0.001
BS = 45

def polynomial_decay(epoch):
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
	return alpha

class History_LAW(Callback):
    def on_train_begin(self, logs={}):
        self.weights_history={}
        self.weights_history["InitialState"]=self.model.weights

    def on_epoch_end(self, epoch, logs={}):
    	modelWeights = {}
    	for layer,i in zip(model.layers,range(len(model.layers))):
    		layerWeights = []
    		for weight in layer.get_weights():
    			layerWeights.append(weight) 
    		modelWeights[str(layer)[str(layer).rindex(".")+1:str(layer).rindex("object")-1]+str(i)]=layerWeights
    	self.weights_history["Epoch"+str(epoch+1)]=(modelWeights)

model_Hist=History_LAW()
mcp_save = ModelCheckpoint('model_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
model = Alexnet.build(10240, 1, 2, reg=0.0002)
opt = SGD(lr=INIT_LR, momentum=0.9)

# opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
# define our set of callbacks and fit the model
callbacks = [mcp_save,LearningRateScheduler(polynomial_decay),model_Hist]

for i in range(config.k):
	for j,k in zip(X.keys(),Y.keys()):
		X_train[j], X_test[j], Y_train[k], Y_test[k] = train_test_split(X[j], Y[k], train_size=config.TRAIN_SPLIT, random_state=i)
	
	for j,k in zip(X_train.keys(),Y_train.keys()):
		X_train[j], X_val[j], Y_train[k], Y_val[k] = train_test_split(X_train[j], Y_train[k], test_size=config.VAL_SPLIT, random_state=i)
	arrays=[X_train[_] for _ in X_train.keys()]
	X_train_DS=np.concatenate(arrays, axis=0)
	X_train_DS=np.reshape(X_train_DS,(X_train_DS.shape[0],X_train_DS.shape[1],1))

	arrays=[X_val[_] for _ in X_val.keys()]
	X_val_DS=np.concatenate(arrays, axis=0)
	X_val_DS=np.reshape(X_val_DS,(X_val_DS.shape[0],X_val_DS.shape[1],1))
	
	arrays=[Y_val[_] for _ in Y_val.keys()]
	Y_val_DS=np.concatenate(arrays, axis=0)
	Y_val_DS=np.reshape(Y_val_DS,(Y_val_DS.shape[0],Y_val_DS.shape[1]))
	
	arrays=[Y_train[_] for _ in Y_train.keys()]
	Y_train_DS=np.concatenate(arrays, axis=0)
	Y_train_DS=np.reshape(Y_train_DS,(Y_train_DS.shape[0],Y_train_DS.shape[1]))
	
	print('\nFold ', i) 
	totalTrain=len(X_train_DS)
	totalVal=len(X_val_DS)
	H = model.fit(X_train_DS,
				Y_train_DS,
				batch_size=BS,
				steps_per_epoch=totalTrain // BS,
				validation_data=(X_val_DS,Y_val_DS),
				validation_steps=totalVal // BS,
				epochs=NUM_EPOCHS,
				callbacks=callbacks,
				use_multiprocessing=True,
                workers=config.WORKERS)
	savemat("weights.mat", model_Hist.weights_history, oned_as='row')
	df.from_dict(H.history).to_csv("H"+ str(i) +".csv",index=False)
	
	# plot the training  loss and accuracy
	N = NUM_EPOCHS
 # plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.title("Training and Validation Loss on Dataset")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend(loc="upper right")
	plt.savefig("Loss"+str(i)+".png") 
 
 # plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training and Validation Accuracy on Dataset")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend(loc="lower right") 
	plt.savefig("accuracy"+str(i)+".png")
	model_json = model.to_json(indent=3)
	with open("architecture.json", "w") as json_file:
		json_file.write(model_json)
	save_model(model, "model.hp5", save_format="h5")

	

