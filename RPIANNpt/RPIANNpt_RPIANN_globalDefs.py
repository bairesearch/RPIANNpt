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

useRecursiveLayers = False	#orig: True
useClassificationLayerLoss = False #orig: True		#if false; only use embedding layer loss calculated by reverse projection from target layer	#if true; it uses backprop calculations from target layer (albeit does not modify weights of target layer)
numberOfLayersLow = True	#orig: False	#use 1 FF layer

numberOfSublayers = 1	#default: 1
if(numberOfSublayers > 1):
	subLayerHiddenDimMultiplier = 2

#dataset parameters:
useImageDataset = False 	#not currently supported	#use CIFAR-10 dataset with CNN
if(useImageDataset):
	useTabularDataset = False
	useCNNlayers = True		
else:
	useTabularDataset = True
	useCNNlayers = False
		
#CNN parameters:
inputProjectionActivationFunction = False
if(useImageDataset):
	hiddenLayerSize = 2048	#2048
	if(useCNNlayers):
		numberOfConvlayers = 1	#default: 1	#1 or 2 (untrained projection)
		numberOfLayers = 9 #number FF layers (overridden by numberOfLayersLow)
		inputProjectionActivationFunction = True	#default: False	# The rest of the model (including the frozen target projection and recursive layers) assumes a zero‑mean, roughly symmetric embedding, which the linear random projection preserves. Dropping in a ReLU after each pooling step skews the distribution positive and throws away sign information, so you lose the Johnson–Lindenstrauss style properties you want from an untrained projection. 
	#	CNNmaxPool = False
	#	CNNbatchNorm = False

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
