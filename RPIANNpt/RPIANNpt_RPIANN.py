"""RPIANNpt_RPIANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
RPIANNpt Recursive Prediction Improvement artificial neural network

"""

from ANNpt_globalDefs import *
from torchsummary import summary
import RPIANNpt_RPIANNmodel
import ANNpt_data

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)

	if(useImageDataset):
		inputImageShape = CNNinputShape
	else:
		inputImageShape = None
	if(printRPIANNmodelProperties):
		print("Creating new model:")
		print("\t ---")
		print("\t datasetType = ", datasetType)
		print("\t stateTrainDataset = ", stateTrainDataset)
		print("\t stateTestDataset = ", stateTestDataset)
		print("\t ---")
		print("\t datasetName = ", datasetName)
		print("\t datasetRepeatSize = ", datasetRepeatSize)
		print("\t trainNumberOfEpochs = ", trainNumberOfEpochs)
		print("\t ---")
		print("\t batchSize = ", batchSize)
		print("\t numberOfLayers = ", numberOfLayers)
		print("\t hiddenLayerSize = ", hiddenLayerSize)
		print("\t inputLayerSize (numberOfFeatures) = ", numberOfFeatures)
		print("\t outputLayerSize (numberOfClasses) = ", numberOfClasses)
		print("\t ---")
		print("\t trainLocal = ", trainLocal)
		print("\t useClassificationLayerLoss = ", useClassificationLayerLoss)
		print("\t ---")
		print("\t useRecursiveLayers = ", useRecursiveLayers)
		print("\t layersFeedConcatInput = ", layersFeedConcatInput)
		print("\t layersFeedResidualInput = ", layersFeedResidualInput)
		print("\t layerScale = ", layerScale)
		print("\t ---")
		print("\t numberOfSublayers = ", numberOfSublayers)
		if(numberOfSublayers > 1):
			print("\t subLayerHiddenDimMultiplier = ", subLayerHiddenDimMultiplier)
			print("\t subLayerFirstNotTrained = ", subLayerFirstNotTrained)
		print("\t ---")
		print("\t inputProjectionActivationFunction = ", inputProjectionActivationFunction)
		print("\t inputProjectionActivationFunctionTanh = ", inputProjectionActivationFunctionTanh)
		print("\t hiddenActivationFunction = ", hiddenActivationFunction)
		print("\t hiddenActivationFunctionTanh = ", hiddenActivationFunctionTanh)
		print("\t targetProjectionActivationFunction = ", targetProjectionActivationFunction)
		print("\t targetProjectionActivationFunctionTanh = ", targetProjectionActivationFunctionTanh)
		print("\t ---")
		print("\t useImageDataset = ", useImageDataset)
		if(useImageDataset):
			print("\t useCNNlayers = ", useCNNlayers)
			print("\t numberOfConvlayers = ", numberOfConvlayers)
			print("\t imageProjectionActivationFunction = ", imageProjectionActivationFunction)
		#print("\t numberOfConvlayers = ", numberOfConvlayers)


	config = RPIANNpt_RPIANNmodel.RPIANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		numberOfConvlayers = numberOfConvlayers,
		hiddenLayerSize = hiddenLayerSize,
		CNNhiddenLayerSize = CNNhiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		datasetSize = datasetSize,
		numberOfClassSamples = numberOfClassSamples,
		inputImageShape = inputImageShape,
	)
	model = RPIANNpt_RPIANNmodel.RPIANNmodel(config)
		
	print(model)
	#summary(model, input_size=(3, 32, 32))  # adjust input_size as needed

	return model
	
