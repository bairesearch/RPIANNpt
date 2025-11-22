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
debugPrintConcatWeights = False	#requires useLovelyTensors

#dataset parameters:
useImageDataset = False 	#use CIFAR-10 dataset with CNN

#backprop parameters:
trainLocal = True	#default: True #disable for debug/benchmark against standard full layer backprop
useClassificationLayerLoss = False #default: False	#orig: True		#if false; only use embedding layer loss calculated by reverse projection from target layer	#if true; it uses backprop calculations from target layer (albeit does not modify weights of target layer)
useClassificationLayerLossStrict = False	#default: False	#orig: False #if false: use a combination of classification layer loss and embedding layer loss*embedding loss weight[0.1]
trainClassificationLayer = False	#default: False	#orig: False	#requires useClassificationLayerLoss

#recursion parameters:
useRecursiveLayers = True		#default: True	#orig: True
layersFeedConcatInput = True	#default: False	#orig: True
layersFeedResidualInput = False	#default: False	#orig: True
layerScale = 0.25	#default: 0.25	#orig: 0.25	#could be made dependent on layersFeedConcatInput, layersFeedResidualInput, hiddenActivationFunctionTanh etc
initialiseZzero = False
if(layersFeedConcatInput):
	initialiseZzero = True	#default: True	#orig: False
trainFinalIterationOnly = False	#default: False	#orig: False

#FF parameters:
numberOfLayersLow = False	#default: False	#orig: False	#use 1 FF layer

#sublayer parameters:
numberOfSublayers = 1	#default: 1
subLayerFirstMixXembedZStreamsSeparately = False	#initialise (dependent vars)
subLayerFirstSparse = False	#initialise (dependent vars)
if(numberOfSublayers > 1):
	subLayerHiddenDimMultiplier = 2	#default: 2
	subLayerFirstNotTrained = True	#default: True	#orig: False	#first sublayer is untrained random projection (no multilayer backprop)
	if(layersFeedConcatInput):
		subLayerFirstMixXembedZStreamsSeparately = False	#orig: False
	subLayerFirstSparse = False	#default: False	#orig: False	#initialise first sublayer weights with sparse connectivity when untrained	#incomplete
	subLayerFirstSparsityLevel = 0.9	#fraction of first sublayer weights zeroed when subLayerFirstSparse=True (0.0-1.0)
	
#RPICNN parameters:
if(useImageDataset):
	useTabularDataset = False
	useRPICNN = False	#orig: False	#recursive CNN layers
	if(useRPICNN):
		RPICNNuniqueWeightsPerPixel = True	#default: True	#orig: False	#each pixel of the RPICNN action layer has its own unique CNN kernel weights
		if(RPICNNuniqueWeightsPerPixel):
			RPICNNpool = False	#default: False
		else:
			RPICNNpool = True	#default: True
		useCNNinputProjection = False	#default: False	#untrained CNN layers (image projection) - useImageProjection	#when False, RPICNN receives raw image channels as x_embed
		useInputProjection = useCNNinputProjection	#linear input projection is currently not supported by useRPICNN
		useCNNtargetProjection = False	#orig: False	#required to retain target image space 
		targetProjectionExemplarImage = useCNNtargetProjection	#default: useCNNtargetProjection	#orig: False	#required to retain target image space
		assert not (numberOfSublayers > 1 and subLayerFirstNotTrained), "useRPICNN numberOfSublayers>1 does not currently support subLayerFirstNotTrained"
	else:
		RPICNNuniqueWeightsPerPixel = False
		useInputProjection = True	#default: True	#orig: useInputProjection=useCNNinputProjection
		useCNNinputProjection = True	#default: True	#untrained CNN layers (image projection) - useImageProjection
		useCNNtargetProjection = False	#default: False (retaining image space provides no benefit to MLP action layers) #orig: False
		targetProjectionExemplarImage = useCNNtargetProjection	#default: False	(retaining image space provides no benefit to MLP action layers) #orig: False
else:
	useTabularDataset = True
	useRPICNN = False
	useInputProjection = True	#default: True
	useCNNinputProjection = False
	useCNNtargetProjection = False
	targetProjectionExemplarImage = False

#activation function parameters:
inputProjectionActivationFunction = True	#default: True	#orig: False	#relu	#not necessary (removes 50% bits from input projection output), but keeps all layer inputs (x embed and Z) in similar range (ie zero or positive)
inputProjectionActivationFunctionTanh = True	#default: True	#orig: True	#tanh	#input should already be normalised from 0 to 1
hiddenActivationFunction = True	#default: True	#orig: True	#relu
hiddenActivationFunctionTanh = True	#default: True	#orig: True	#tanh
targetProjectionActivationFunction = hiddenActivationFunction	#default: True	#orig: True	#relu
targetProjectionActivationFunctionTanh = hiddenActivationFunctionTanh	#default: True	#orig: False	#tanh
targetProjectionUniquePerLayer = False	#default: False	#orig: False

#projection autoencoder parameters:
useProjectionAutoencoder = False	#default: False	#input/target projection trained using a pseudo-autoencoder algorithm
useProjectionAutoencoderIndependent = False	#default: False	#each direction of the pseudo-autoencoders is trained independently by holding all other weights constant
if(useProjectionAutoencoder):
	inputProjectionAutoencoder = True	#default: False	#input projection trained using a pseudo-autoencoder algorithm	#not compatible with useImageDataset:useCNNinputProjection
	targetProjectionAutoencoder = True	#default: False	#target projection trained using a pseudo-autoencoder algorithm	#not compatible with useImageDataset:useCNNtargetProjection
	if(useProjectionAutoencoderIndependent):
		inputProjectionAutoencoderIndependent = True	#default: False	#each direction of the pseudo-autoencoders is trained independently by holding all other weights constant
		targetProjectionAutoencoderIndependent = True	#default: False	#each direction of the pseudo-autoencoders is trained independently by holding all other weights constant
	else:
		inputProjectionAutoencoderIndependent = False
		targetProjectionAutoencoderIndependent = False

if(useProjectionAutoencoder):
	projectionAutoencoderPretrainEpochs = 10	#default: 10	#orig: 0	#run projection autoencoders in a separate pretraining stage before the main task (0 disables pretrain, >0 disables warmup and run every epoch)
	projectionAutoencoderWarmupEpochs = 0	#default: 0	#orig: 5	#requires trainNumberOfEpochsHigh(~*2)	#run projection autoencoders only for the first N epochs (0 disables warmup and runs every epoch)
	projectionAutoencoderDenoisingStd = 0.0	#default: 0.0 
	projectionAutoencoderVICReg = False
	projectionAutoencoderVICRegContrastiveLoss = False
	if(useProjectionAutoencoderIndependent):
		projectionAutoencoderIndependentSeparateTrainPhases = True	#default: True	orig: False
		if(not projectionAutoencoderIndependentSeparateTrainPhases):
			projectionAutoencoderVICReg = True	#default: False	#apply VICReg constraints to projection embeddings
			projectionAutoencoderVICRegContrastiveLoss = True	#default: False	#apply contrastive loss to projection embeddings
	else:
		projectionAutoencoderIndependentSeparateTrainPhases = False
	if(projectionAutoencoderVICReg):
		projectionAutoencoderVICRegLambda = 25.0
		projectionAutoencoderVICRegMu = 25.0
		projectionAutoencoderVICRegNu = 1.0
		projectionAutoencoderVICRegEps = 1e-4
	if(projectionAutoencoderVICRegContrastiveLoss):
		projectionAutoencoderVICRegContrastiveWeight = 1.0
		projectionAutoencoderVICRegContrastiveMargin = 0.0
	

#target embedding sparsity parameters:
targetProjectionSparse = False #default: False #orig: False	#generate a sparse instead of dense target embedding
targetProjectionSparsityLevel = 0.9	#fraction of target projection weights zeroed when targetProjectionSparse=True (0.0-1.0)

#hidden size parameters:
if(useImageDataset):
	hiddenLayerSize = 2048	#2048	#*8	#default: 2048

#CNN projection parameters:
if(useImageDataset):
	if(useCNNinputProjection or useCNNtargetProjection):
		useCNNprojection = True	#untrained CNN layers (input/target image projection)
	else:
		useCNNprojection = False
	if(useCNNprojection):
		CNNprojectionNumlayers = 1	#default: 1	#1 or 2 (untrained CNN projection)
		if(useRPICNN):
			CNNprojectionStride = 1	#default: 1 (preserves spatial size) #stride for image projection pooling layers	#use stride 1 for RPICNN to use original image spatial dimensions
		else:
			CNNprojectionStride = 2	#default: 2	#stride for image projection pooling layers
		CNNprojectionActivationFunction = True	#default: True	#orig: True	#relu	#will effectively override inputProjectionActivationFunction=False
	else:
		CNNprojectionNumlayers = 1	#effective CNN projection layers (no change from input pixel space)
		CNNprojectionStride = 1	#effective CNN stride (no change from input pixel space)
else:
	useCNNprojection = False

#layer parameters:
if(useImageDataset):
	if(useRPICNN):
		numberOfConvlayers = 6	#number RPICNN layers 	#default: 2, 4, 6	#orig: 9	#default: 6
		numberOfFFLayers = 3	#number FF layers	#orig: 0	#default: 3
		numberOfLayers = numberOfConvlayers+numberOfFFLayers #(overridden by numberOfLayersLow)
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
