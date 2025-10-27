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

#debug parameters:
printRPIANNmodelProperties = True

#backprop parameters:
trainLocal = True	#default: True #disable for debug/benchmark against standard full layer backprop
useClassificationLayerLoss = False #default: False	#orig: True		#if false; only use embedding layer loss calculated by reverse projection from target layer	#if true; it uses backprop calculations from target layer (albeit does not modify weights of target layer)

#recursion parameters:
useRecursiveLayers = True		#orig: True
layersFeedConcatInput = True	#default: False	#orig: True
layersFeedResidualInput = False	#default: False	#orig: True
layerScale = 0.25	#default: 0.25	#orig: 0.25	#could be made dependent on layersFeedConcatInput, layersFeedResidualInput, hiddenActivationFunctionTanh etc
initialiseYhatZero = False
if(layersFeedConcatInput):
	initialiseYhatZero = True	#default: True	#orig: False

#FF parameters:
numberOfLayersLow = False	#orig: False	#use 1 FF layer

#sublayer parameters:
numberOfSublayers = 1	#default: 1
subLayerFirstMixXembedYhatStreamsSeparately = False	#initialise (dependent vars)
subLayerFirstSparse = False	#initialise (dependent vars)
if(numberOfSublayers > 1):
	subLayerHiddenDimMultiplier = 2	#default: 2
	subLayerFirstNotTrained = True	#default: True	#orig: False	#first sublayer is untrained random projection (no multilayer backprop)
	if(layersFeedConcatInput):
		subLayerFirstMixXembedYhatStreamsSeparately = True	#orig: False
	subLayerFirstSparse = False	#default: False	#orig: False	#initialise first sublayer weights with sparse connectivity when untrained	#incomplete
	subLayerFirstSparsityLevel = 0.9	#fraction of first sublayer weights zeroed when subLayerFirstSparse=True (0.0-1.0)
	
#dataset parameters:
useImageDataset = False 	#use CIFAR-10 dataset with CNN
if(useImageDataset):
	useTabularDataset = False
	useCNNlayers = True
else:
	useTabularDataset = True
	useCNNlayers = False
		
#activation function parameters:
inputProjectionActivationFunction = True	#default: True	#orig: False	#relu
inputProjectionActivationFunctionTanh = True	#default: True	#orig: True	#tanh
hiddenActivationFunction = inputProjectionActivationFunction	#default: True	#orig: True	#relu
hiddenActivationFunctionTanh = inputProjectionActivationFunctionTanh	#default: True	#orig: True	#tanh
targetProjectionActivationFunction = hiddenActivationFunction	#default: True	#orig: True	#relu
targetProjectionActivationFunctionTanh = hiddenActivationFunctionTanh	#default: True	#orig: False	#tanh

#CNN parameters:
if(useImageDataset):
	hiddenLayerSize = 2048	#2048
	if(useCNNlayers):
		numberOfConvlayers = 1	#default: 1	#1 or 2 (untrained projection)
		numberOfLayers = 9 #number FF layers (overridden by numberOfLayersLow)
		imageProjectionActivationFunction = True	#default: True	#orig: True	#relu
	#	CNNmaxPool = False
	#	CNNbatchNorm = False

if(useTabularDataset):
	datasetType = "useTabularDataset"
elif(useImageDataset):
	datasetType = "useImageDataset"

#training/network scale parameters:
trainNumberOfEpochsHigh = False	#use ~9x more epochs to train
hiddenLayerSizeHigh = True	#use ~4x more hidden neurons (approx equalise number of parameters with ANN)	#large projection from input/output

#data storage parameters:
workingDrive = '/large/source/ANNpython/RPIANNpt/'
dataDrive = workingDrive	#'/datasets/'
modelName = 'modelRPIANN'
