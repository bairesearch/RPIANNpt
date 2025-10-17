"""RPIANNpt_RPIANN_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
RPIANNpt globalDefs

"""

printRPIANNmodelProperties = True

trainLocal = True	#default: True #disable for debug/benchmark against standard full layer backprop

recursiveActionLayers = 1	#default: 1	#or 2
if(recursiveActionLayers > 1):
	recursiveActionLayerHiddenDimMultiplier = 2

#dataset parameters:
useImageDataset = False 	#not currently supported	#use CIFAR-10 dataset with CNN
if(useImageDataset):
	useTabularDataset = False
	useCNNlayers = False		#not currently supported
else:
	useTabularDataset = True
	useCNNlayers = False	 #default:False	#optional	#enforce different connection sparsity across layers to learn unique features with greedy training	#use 2D CNN instead of linear layers

#CNN parameters:
if(useCNNlayers):
	#create CNN architecture, where network size converges by a factor of ~4 (or 2*2) per layer and number of channels increases by the same factor
	CNNkernelSize = 3
	CNNstride = 1
	CNNpadding = "same"
	useCNNlayers2D = True
	CNNinputWidthDivisor = 2
	CNNinputSpaceDivisor = CNNinputWidthDivisor*CNNinputWidthDivisor
	CNNinputPadding = False
	CNNmaxInputPadding = False	#pad input with zeros such that CNN is applied to every layer
	debugCNN = False
	if(CNNstride == 1):
		CNNmaxPool = True
		#assert not supportSkipLayers, "supportSkipLayers not currently supported with CNNstride == 1 and CNNmaxPool"
	elif(CNNstride == 2):
		CNNmaxPool = False
		assert CNNkernelSize==2
	else:
		print("error: CNNstride>2 not currently supported")
	CNNbatchNorm = True
else:
	CNNmaxPool = False
	CNNbatchNorm = False

if(useTabularDataset):
	datasetType = "useTabularDataset"
elif(useImageDataset):
	datasetType = "useImageDataset"

#training epoch parameters:
trainNumberOfEpochsHigh = False	#use ~4x more epochs to train
hiddenLayerSizeHigh = True	#use ~4x more hidden neurons (approx equalise number of parameters with ANN)	#large projection from input/output

#data storage parameters:
workingDrive = '/large/source/ANNpython/RPIANNpt/'
dataDrive = workingDrive	#'/datasets/'
modelName = 'modelRPIANN'
