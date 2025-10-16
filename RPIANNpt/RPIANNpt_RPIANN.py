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
RPIANNpt_RPIANN Autoencoder/Breakaway generated artificial neural network

"""

from ANNpt_globalDefs import *
from torchsummary import summary
import RPIANNpt_RPIANNmodel
import ANNpt_data

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	
	print("creating new model")
	config = RPIANNpt_RPIANNmodel.AEANNconfig(
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
		numberOfClassSamples = numberOfClassSamples
	)
	model = RPIANNpt_RPIANNmodel.AEANNmodel(config)
	
	print(model)
	#summary(model, input_size=(3, 32, 32))  # adjust input_size as needed

	return model
	
