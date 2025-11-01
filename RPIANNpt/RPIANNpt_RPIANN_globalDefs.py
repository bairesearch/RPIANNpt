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
useRecursiveLayers = True		#default: True	#orig: True
layersFeedConcatInput = True	#default: False	#orig: True
layersFeedResidualInput = False	#default: False	#orig: True
layerScale = 0.25	#default: 0.25	#orig: 0.25	#could be made dependent on layersFeedConcatInput, layersFeedResidualInput, hiddenActivationFunctionTanh etc
initialiseYhatZero = False
if(layersFeedConcatInput):
	initialiseYhatZero = True	#default: True	#orig: False

#FF parameters:
numberOfLayersLow = False	#default: False	#orig: False	#use 1 FF layer

#sublayer parameters:
numberOfSublayers = 1	#default: 1
subLayerFirstMixXembedYhatStreamsSeparately = False	#initialise (dependent vars)
subLayerFirstSparse = False	#initialise (dependent vars)
if(numberOfSublayers > 1):
	subLayerHiddenDimMultiplier = 2	#default: 2
	subLayerFirstNotTrained = True	#default: True	#orig: False	#first sublayer is untrained random projection (no multilayer backprop)
	if(layersFeedConcatInput):
		subLayerFirstMixXembedYhatStreamsSeparately = False	#orig: False
	subLayerFirstSparse = False	#default: False	#orig: False	#initialise first sublayer weights with sparse connectivity when untrained	#incomplete
	subLayerFirstSparsityLevel = 0.9	#fraction of first sublayer weights zeroed when subLayerFirstSparse=True (0.0-1.0)
	
#dataset parameters:
useImageDataset = False 	#use CIFAR-10 dataset with CNN
if(useImageDataset):
	useTabularDataset = False
	useRPICNN = False	#orig: False	#recursive CNN layers
	if(useRPICNN):
		RPICNNuniqueWeightsPerPixel = True	#default: True	#orig: False	#each pixel of the RPICNN action layer has its own unique CNN kernel weights
		useCNNlayersInputProjection = False	#default: False	#untrained CNN layers (image projection) - useImageProjection	#when False, RPICNN receives raw image channels as x_embed
		useCNNlayersTargetProjection = False	#orig: False	#required to retain target image space 
		targetProjectionExemplarImage = useCNNlayersTargetProjection	#orig: False	#required to retain target image space
		assert not (numberOfSublayers > 1 and subLayerFirstNotTrained), "useRPICNN numberOfSublayers>1 does not currently support subLayerFirstNotTrained"
	else:
		useCNNlayersInputProjection = True	#mandatory: True	#untrained CNN layers (image projection) - useImageProjection
		useCNNlayersTargetProjection = False	#default: False (retaining image space provides no benefit to MLP action layers) #orig: False
		targetProjectionExemplarImage = useCNNlayersTargetProjection	#default: False	(retaining image space provides no benefit to MLP action layers) #orig: False
else:
	useTabularDataset = True
	useRPICNN = False
	useCNNlayers = False
	useCNNlayersInputProjection = False
	useCNNlayersTargetProjection = False
	targetProjectionExemplarImage = False

#activation function parameters:
inputProjectionActivationFunction = True	#default: True	#orig: False	#relu	#not necessary (removes 50% bits from input projection output), but keeps all layer inputs (x embed and y hat) in similar range (ie zero or positive)
inputProjectionActivationFunctionTanh = True	#default: True	#orig: True	#tanh	#input should already be normalised from 0 to 1
hiddenActivationFunction = True	#default: True	#orig: True	#relu
hiddenActivationFunctionTanh = True	#default: True	#orig: True	#tanh
targetProjectionActivationFunction = hiddenActivationFunction	#default: True	#orig: True	#relu
targetProjectionActivationFunctionTanh = hiddenActivationFunctionTanh	#default: True	#orig: False	#tanh

#CNN parameters:
if(useImageDataset):
	hiddenLayerSize = 2048	#default: 2048
	if(useCNNlayersInputProjection or useCNNlayersTargetProjection):
		useCNNlayers = True	#untrained CNN layers (input/target image projection)
	else:
		useCNNlayers = False
	if(useCNNlayers):
		numberOfConvlayers = 1	#default: 1	#1 or 2 (untrained CNN projection)
		if(useRPICNN):
			CNNprojectionStride = 1	#default: 1 (preserves spatial size) #stride for image projection pooling layers	#use stride 1 for RPICNN to use original image spatial dimensions
		else:
			CNNprojectionStride = 2	#default: 2	#stride for image projection pooling layers
		imageProjectionActivationFunction = True	#default: True	#orig: True	#relu	#will effectively override inputProjectionActivationFunction=False
	else:
		CNNprojectionStride = 1	#effective CNN stride (no change from input pixel space)
	if(useRPICNN):
		numberOfLayers = 9 #number RPICNN layers (overridden by numberOfLayersLow)	#default: 9
	else:
		numberOfLayers = 9 #number FF layers (overridden by numberOfLayersLow)	#default: 9

if(useTabularDataset):
	datasetType = "useTabularDataset"
elif(useImageDataset):
	datasetType = "useImageDataset"

#training/network scale parameters:
trainRepeatBatchX = 1	#default: 1	#trains each batch ~9x	#temp
trainNumberOfEpochsHigh = False	#default: False	#use ~4x more epochs to train
hiddenLayerSizeHigh = True	#default: True	#use ~4x more hidden neurons	#large projection from input/output

#data storage parameters:
workingDrive = '/large/source/ANNpython/RPIANNpt/'
dataDrive = workingDrive	#'/datasets/'
modelName = 'modelRPIANN'
