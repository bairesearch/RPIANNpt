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
import torch as pt
import RPIANNpt_RPIANNmodel
import ANNpt_data


def _extract_class_exemplar_images(dataset, number_of_classes):
	if not targetProjectionExemplarImage or not useImageDataset:
		return None
	try:
		import torchvision.transforms as transforms
		from PIL import Image
	except ImportError as exc:
		raise ImportError("Torchvision and PIL are required to extract exemplar images.") from exc

	base_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	exemplars = [None] * number_of_classes
	remaining = set(range(number_of_classes))

	data_attr = getattr(dataset, "data", None)
	targets_attr = getattr(dataset, "targets", None)

	if data_attr is not None and targets_attr is not None:
		for raw_image, label in zip(data_attr, targets_attr):
			label_idx = int(label)
			if label_idx in remaining:
				pil_image = Image.fromarray(raw_image)
				tensor_image = base_transform(pil_image)
				exemplars[label_idx] = tensor_image
				remaining.discard(label_idx)
				if not remaining:
					break
	else:
		for index in range(len(dataset)):
			image, label = dataset[index]
			label_idx = int(label)
			if label_idx in remaining:
				if isinstance(image, pt.Tensor):
					tensor_image = image.clone()
				else:
					pil_image = Image.fromarray(image)
					tensor_image = base_transform(pil_image)
				exemplars[label_idx] = tensor_image
				remaining.discard(label_idx)
				if not remaining:
					break

	if any(exemplar is None for exemplar in exemplars):
		raise ValueError("Unable to locate exemplar images for all classes in the training dataset.")

	return pt.stack(exemplars)

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)

	if(useImageDataset):
		inputImageShape = CNNinputShape
	else:
		inputImageShape = None

	if(targetProjectionExemplarImage and useImageDataset):
		class_exemplar_images = _extract_class_exemplar_images(dataset, numberOfClasses)
	else:
		class_exemplar_images = None
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
		print("\t useClassificationLayerLossStrict = ", useClassificationLayerLossStrict)
		print("\t trainClassificationLayer = ", trainClassificationLayer)
		print("\t ---")
		print("\t useRecursiveLayers = ", useRecursiveLayers)
		print("\t layersFeedConcatInput = ", layersFeedConcatInput)
		print("\t layersFeedResidualInput = ", layersFeedResidualInput)
		print("\t layerScale = ", layerScale)
		print("\t initialiseZzero = ", initialiseZzero)
		print("\t trainFinalIterationOnly = ", trainFinalIterationOnly)
		print("\t ---")
		print("\t numberOfSublayers = ", numberOfSublayers)
		if(numberOfSublayers > 1):
			print("\t subLayerHiddenDimMultiplier = ", subLayerHiddenDimMultiplier)
			print("\t subLayerFirstNotTrained = ", subLayerFirstNotTrained)
			print("\t subLayerFirstMixXembedZStreamsSeparately = ", subLayerFirstMixXembedZStreamsSeparately)
			print("\t subLayerFirstSparse = ", subLayerFirstSparse)
			print("\t subLayerFirstSparsityLevel = ", subLayerFirstSparsityLevel)
		print("\t ---")
		print("\t inputProjectionActivationFunction = ", inputProjectionActivationFunction)
		print("\t inputProjectionActivationFunctionTanh = ", inputProjectionActivationFunctionTanh)
		print("\t hiddenActivationFunction = ", hiddenActivationFunction)
		print("\t hiddenActivationFunctionTanh = ", hiddenActivationFunctionTanh)
		print("\t targetProjectionActivationFunction = ", targetProjectionActivationFunction)
		print("\t targetProjectionActivationFunctionTanh = ", targetProjectionActivationFunctionTanh)
		print("\t targetProjectionUniquePerLayer = ", targetProjectionUniquePerLayer)
		print("\t ---")
		print("\t targetProjectionSparse = ", targetProjectionSparse)
		print("\t targetProjectionSparsityLevel = ", targetProjectionSparsityLevel)
		print("\t ---")
		print("\t useImageDataset = ", useImageDataset)
		if(useImageDataset):
			print("\t targetProjectionExemplarImage = ", targetProjectionExemplarImage)
			print("\t useCNNprojection = ", useCNNprojection)
			if(useCNNprojection):
				print("\t\t useCNNinputProjection = ", useCNNinputProjection)
				print("\t\t useCNNtargetProjection = ", useCNNtargetProjection)
				print("\t\t CNNprojectionNumlayers = ", CNNprojectionNumlayers)
				print("\t\t CNNprojectionStride = ", CNNprojectionStride)
				print("\t\t CNNprojectionActivationFunction = ", CNNprojectionActivationFunction)
			print("\t useRPICNN = ", useRPICNN)
			if(useRPICNN):
				print("\t\t numberOfConvlayers = ", numberOfConvlayers)
				print("\t\t numberOfFFLayers = ", numberOfFFLayers)
				print("\t\t RPICNNuniqueWeightsPerPixel = ", RPICNNuniqueWeightsPerPixel)
				print("\t\t RPICNNpool = ", RPICNNpool)


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
		class_exemplar_images = class_exemplar_images,
	)
	model = RPIANNpt_RPIANNmodel.RPIANNmodel(config)
		
	print(model)
	#summary(model, input_size=(3, 32, 32))  # adjust input_size as needed

	return model
	
