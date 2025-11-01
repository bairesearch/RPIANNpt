"""ANNpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt globalDefs

"""


#algorithm selection
useAlgorithmVICRegANN = False
useAlgorithmAUANN = False
useAlgorithmLIANN = False
useAlgorithmLUANN = False
useAlgorithmLUOR = False
useAlgorithmSANIOR = False
useAlgorithmEIANN = False
useAlgorithmEISANI = False
useAlgorithmAEANN = False
useAlgorithmRPIANN = True
useAlgorithmANN = False

#train/test vars;
stateTrainDataset = True
stateTestDataset = True

saveModelTrain = True
saveModelTrainContinuous = True

#cloud execution;
useCloudExecution = False	#jupyter notebook does not support long cmd output
if(useCloudExecution):
	relativeFolderLocations = True
	debugPrintGPUusage = True
else:
	relativeFolderLocations = False
	debugPrintGPUusage = False
	
#initialise (dependent vars);
usePairedDataset = False
datasetNormalise = False
datasetRepeat = False
datasetRepeatEpochModifier = 'none'	#default: 'none'
datasetRepeatSize = 1
datasetShuffle = False	#automatically performed by generateVICRegANNpairedDatasets
datasetOrderByClass = False	#automatically performed by generateVICRegANNpairedDatasets
dataloaderShuffle = True
dataloaderMaintainBatchSize = True
dataloaderRepeat = False
dataloaderNumWorkers = 4
dataloaderPinMemory = True

optimiserAdam = True

#initialise (dependent vars);
useCustomWeightInitialisation = False
useCustomBiasInitialisation = False
useCustomBiasNoTrain = False
useSignedWeights = False
usePositiveWeightsClampModel = False
useCPU = False

debugSmallBatchSize = False	#small batch size for debugging matrix output
debugSmallDataset = False	#small dataset (no datasetRepeat) for debugging matrix output
debugDataNormalisation = False
debugOnlyPrintStreamedWikiArticleTitles = False

trainLocal = False
trainGreedy = False
trainIndividialSamples = False

useLinearSublayers = False	#use multiple independent sublayers per linear layer	#optional
if(useLinearSublayers):
	linearSublayersNumber = 10
else:
	linearSublayersNumber = 1
	

#default network hierarchy parameters (overwritten by specific dataset defaults): 
warmupEpochs = 0	#default: 0
learningRate = 0.005	#0.005	#0.0001
momentum = 0.0     #default: 0.0
weightDecay  = 0.0 #default: 0.0
batchSize = 128	 #default: 128	#orig: 64	#debug: 2
numberOfLayers = 4	#default: 4	#counts hidden and output layers (not input layer)
hiddenLayerSize = 10	#default: 10
trainNumberOfEpochs = 10	#default: 10
numberOfConvlayers = None
CNNhiddenLayerSize = None
useLearningRateScheduler = False
batchNormFC = False	#optional	#batch norm for fully connected layers
dropout = False
dropoutProb = 0.5 	#default: 0.5	#orig: 0.3

#initialise (dependent vars);
debugSmallNetwork = False
trainNumberOfEpochsHigh = False
hiddenLayerSizeHigh = False
numberOfLayersLow = False
inputLayerInList = True
outputLayerInList = True
useCNNlayers = False
useCNNlayers2D = True
thresholdActivations = False
debugPrintActivationOutput = False
simulatedDendriticBranches = False
activationFunctionType = "relu"
trainLastLayerOnly = False
normaliseActivationSparsity = False
debugUsePositiveWeightsVerify = False
datasetNormaliseMinMax = True	#normalise between 0.0 and 1.0
datasetNormaliseStdAvg = False	#normalise based on std and mean (~-1.0 to 1.0)
supportSkipLayers = False
supportSkipLayersResidual = False
supportFieldTypeList = False

useInbuiltCrossEntropyLossFunction = True	#required
if(useSignedWeights):
	usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required


#initialise (dependent vars);
datasetReplaceNoneValues = False
datasetConvertClassValues = False	#reformat class values from 0.. ; contiguous (will also convert string to int)
datasetConvertFeatureValues = False	#reformat feature values from 0.. ; contiguous (will also convert string to int)
datasetLocalFile = False
datasetSpecifyDataFiles = True	#specify data file names in dataset (else automatically selected by huggingface)
datasetHasTestSplit = True
datasetHasSubsetType = False
datasetEqualiseClassSamples = False
datasetEqualiseClassSamplesTest = False 
disableDatasetCache = False

datasetLocalFileOptimise = False
datasetCorrectMissingValues = False	
datasetConvertClassTargetColumnFloatToInt = False
dataloaderRepeatSampler = False	
dataloaderRepeatLoop = False		#legacy (depreciate)
debugCullDatasetSamples = False

#import algorithm specific globalDefs;
useTabularDataset = False
useImageDataset = False
useNLPDataset = False
if(useAlgorithmVICRegANN):
	from VICRegANNpt_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmAUANN):
	from LREANNpt_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmLIANN):
	from LIANNpt_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmLUANN):
	from LUANNpt_LUANN_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmLUOR):
	from LUANNpt_LUOR_globalDefs import *
	useImageDataset = True
elif(useAlgorithmSANIOR):
	from LUANNpt_SANIOR_globalDefs import *
	useImageDataset = True
elif(useAlgorithmEIANN):
	from EIANNpt_EIANN_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmEISANI):
	from EISANIpt_EISANI_globalDefs import *
	#useTabularDataset/useImageDataset is defined by EISANIpt_EISANI_globalDefs
elif(useAlgorithmAEANN):
	from AEANNpt_AEANN_globalDefs import *
	#useTabularDataset/useImageDataset is defined by AEANNpt_AEANN_globalDefs
elif(useAlgorithmRPIANN):
	from RPIANNpt_RPIANN_globalDefs import *
	#useTabularDataset/useImageDataset is defined by RPIANNpt_RPIANN_globalDefs
elif(useAlgorithmANN):
	from ANNpt_ANN_globalDefs import *
	#useTabularDataset/useImageDataset is defined by ANNpt_ANN_globalDefs
	
if(useCustomWeightInitialisation):
	Wmean = 0.0
	WstdDev = 0.05	#stddev of weight initialisations

if(useTabularDataset):
	#datasetName = 'tabular-benchmark'	#expected test accuracy: ~63%
	#datasetName = 'blog-feedback'	#expected test accuracy: ~70%
	datasetName = 'titanic'	#expected test accuracy: ~90%
	#datasetName = 'red-wine'	#expected test accuracy: ~58%
	#datasetName = 'breast-cancer-wisconsin'	#expected test accuracy: ~91%
	#datasetName = 'diabetes-readmission'	#expected test accuracy: ~61%
	#datasetName = 'banking-marketing'	#expected test accuracy: ~85%	#third party benchmark accuracy: ~79.1%
	#datasetName = 'adult_income_dataset'	#expected test accuracy: ~86% 	#third party benchmark accuracy: ~85.8%
	#datasetName = 'covertype'	#expected test accuracy: ~%92 (10 epochs) 	#third party benchmark accuracy: ~97.1%
	#datasetName = 'higgs'	#expected test accuracy: ~70%	#third party benchmark accuracy: 73.8%	#https://archive.ics.uci.edu/dataset/280/higgs
	#datasetName = 'topquark'	#expected test accuracy: ~85%
	#datasetName = 'iris'	#expected test accuracy: ~95%
	#datasetName = 'new-thyroid'	#expected test accuracy: ~95%	https://archive.ics.uci.edu/dataset/102/thyroid+disease
	if(datasetName == 'tabular-benchmark'):
		datasetNameFull = 'inria-soda/tabular-benchmark'
		classFieldName = 'class'
		trainFileName = 'clf_cat/albert.csv'
		datasetHasTestSplit = False
		datasetNormalise = True
		#datasetShuffle = True	#raw dataset is not shuffled	#not required with dataloaderShuffle 
		learningRate = 0.001
		hiddenLayerSize = 64	#default: 64	#orig: 10
		numberOfLayers = 4	#default: 4
		numberOfSamplesK = 58.3
	elif(datasetName == 'blog-feedback'):
		datasetNameFull = 'wwydmanski/blog-feedback'
		classFieldName = 'target'
		datasetSpecifyDataFiles = False
		datasetNormalise = True
		#datasetConvertClassValues = True	#int: not contiguous	#alternate method (slower)
		datasetConvertClassTargetColumnFloatToInt = True	#int: not contiguous
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 144	#default: 144	#orig: 144	#old 800	#144=288/2 [input features padded / 2]
		#datasetEqualiseClassSamples = False	#extremely imbalanced classes
		numberOfSamplesK = 52.4
	elif(datasetName == 'titanic'):
		datasetNameFull = 'victor/titanic'
		classFieldName = '2urvived'
		datasetSpecifyDataFiles = False
		datasetReplaceNoneValues = True
		datasetNormalise = True
		datasetCorrectMissingValues = True
		#datasetShuffle = True	#raw dataset is not completely shuffled	#not required with dataloaderShuffle 
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 128	#default: 128	#orig: 100
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10
		numberOfSamplesK = 1.3
	elif(datasetName == 'red-wine'):
		datasetNameFull = 'lvwerra/red-wine'
		classFieldName = 'quality'
		trainFileName = 'winequality-red.csv'
		datasetHasTestSplit = False
		datasetConvertClassValues = True	#int: not start at 0
		datasetNormalise = True
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4	#external benchmark: 4
		hiddenLayerSize = 128	#default: 128	#orig: 100	#external benchmark; 64/128
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10
		numberOfSamplesK = 1.6
	elif(datasetName == 'breast-cancer-wisconsin'):
		datasetNameFull = 'scikit-learn/breast-cancer-wisconsin'
		classFieldName = 'diagnosis'
		trainFileName = 'breast_cancer.csv'
		datasetHasTestSplit = False
		datasetReplaceNoneValues = True
		datasetConvertClassValues = True	#string: B/M
		datasetNormalise = True
		datasetCorrectMissingValues = True
		#datasetShuffle = True	#raw dataset is not completely shuffled	#not required with dataloaderShuffle 
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 32	#default: 32	#orig: 20	#old: 100
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10
		numberOfSamplesK = 0.569
	elif(datasetName == 'diabetes-readmission'):
		datasetNameFull = 'imodels/diabetes-readmission'
		classFieldName = 'readmitted'
		datasetSpecifyDataFiles = False
		datasetNormalise = True
		learningRate = 0.005
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 304	#default: 304	#orig: 10, 256
		numberOfSamplesK = 81.4
	elif(datasetName == 'banking-marketing'):
		datasetSpecifyDataFiles = False
		datasetConvertFeatureValues = True	#required if they contain strings
		datasetNameFull = 'Andyrasika/banking-marketing'
		classFieldName = 'y'
		datasetConvertClassValues = True	#string: yes/no
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 5	#default: 5
		hiddenLayerSize = 128	#default: 128
		numberOfSamplesK = 45.2
	elif(datasetName == 'adult_income_dataset'):
		datasetSpecifyDataFiles = False
		datasetConvertFeatureValues = True	#required if they contain strings
		datasetHasTestSplit = False
		datasetNameFull = 'meghana/adult_income_dataset'
		classFieldName = 'income'
		datasetConvertClassValues = True	#string: <=50K/>50K
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 256	#default: 256
		numberOfSamplesK = 48.8
	elif(datasetName == 'covertype'):
		datasetSpecifyDataFiles = False
		datasetConvertFeatureValues = True	#required if they contain strings
		datasetHasTestSplit = False
		datasetHasSubsetType = True
		datasetSubsetName = 'covertype'
		datasetNameFull = 'mstz/covertype'
		classFieldName = 'cover_type'
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 6	#default: 6
		hiddenLayerSize = 512	#default: 512
		numberOfSamplesK = 581
	elif(datasetName == 'higgs'):
		datasetLocalFile = True		#manually download higgs.zip:HIGGS.csv from https://archive.ics.uci.edu/dataset/280/higgs
		#echo "class,lepton_pT,lepton_eta,lepton_phi,missing_energy_magnitude,missing_energy_phi,jet1pt,jet1eta,jet1phi,jet1b,jet2pt,jet2eta,jet2phi,jet2b,jet3pt,jet3eta,jet3phi,jet3b,jet4pt,jet4eta,jet4phi,jet4b,m_jj,m_jjj,m_lv,m_jlv,m_bb,m_wbb,m_wwbb" | cat - HIGGS.csv > HIGGSwithHeader.csv
		classFieldName = 'class'
		trainFileName = 'HIGGSwithHeader.csv'
		datasetHasTestSplit = False
		datasetConvertClassTargetColumnFloatToInt = True
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 4
		#numberOfLayers = 2
		hiddenLayerSize = 256	#default: 256
		datasetLocalFileOptimise = False	#default: False
		numberOfSamplesK = 11000
	elif(datasetName == 'new-thyroid'):
		classFieldName = 'class'
		trainFileName = 'new-thyroid.csv'	#manually download thyroid+disease.zip:new-thyroid.data from https://archive.ics.uci.edu/dataset/102/thyroid+disease
		datasetHasTestSplit = False
		#datasetShuffle = True	#raw dataset is not completely shuffled	#not required with dataloaderShuffle 
		datasetLocalFile = True	
		datasetNormalise = True
		datasetConvertClassValues = True
		learningRate = 0.005	#default: 0.005
		numberOfLayers = 3	#default: 3	#orig: 2, 4
		hiddenLayerSize = 16	#default: 16	#orig: 4, 10
		datasetRepeat = True	#enable better sampling by dataloader with high batchSize (required if batchSize ~= datasetSize)
		if(datasetRepeat):
			datasetRepeatSize = 10	#required for batchSize ~= 64
		numberOfSamplesK = 0.215
	elif(datasetName == 'topquark'):
		datasetNameFull = 'lewtun/top_quark_tagging'
		classFieldName = 'is_signal_new'
		datasetSpecifyDataFiles = False
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 5	#default: 5
		hiddenLayerSize = 256	#default: 256
	elif(datasetName == 'iris'):
		datasetNameFull = 'scikit-learn/iris'
		classFieldName = 'Species'
		datasetSpecifyDataFiles = False
		datasetConvertClassValues = True
		datasetNormalise = True
		datasetHasTestSplit = False
		dataloaderMaintainBatchSize = False
		learningRate = 0.005	#default: 0.005
		numberOfLayers = 3	#default: 3
		hiddenLayerSize = 32	#default: 32
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10	#required for batchSize ~= 64
	#elif ...

	if(datasetLocalFileOptimise):
		datasetShuffle = True	#default: True	#required for high dataloader initialisation efficiency with large datasets
		dataloaderShuffle = False	#default: False	#required for high dataloader initialisation efficiency with large datasets
		disableDatasetCache = True	#default: False #requires high CPU ram	#prevents large cache from being created on disk #only suitable with datasetLocalFile
	else:
		disableDatasetCache = False

	if(datasetRepeat):
		if(datasetRepeatEpochModifier == '*'):	#equalise the number of samples between datasets trained
			trainNumberOfEpochs *= datasetRepeatSize
		elif(datasetRepeatEpochModifier == '/'):	#use the original number of samples for each dataset during training
			trainNumberOfEpochs //= datasetRepeatSize
		elif(datasetRepeatEpochModifier == 'None'):
			pass			
	
	if(dataloaderRepeat):
		dataloaderRepeatSize = 10	#number of repetitions
		dataloaderRepeatLoop = False	#legacy (depreciate)
		dataloaderRepeatSampler = True
		if(dataloaderRepeatSampler):
			dataloaderRepeatSamplerCustom = False	#no tqdm visualisation
			assert not dataloaderShuffle	#dataloaderShuffle is not supported by dataloaderRepeatSampler
	
	saveModelTrainContinuous = False
elif(useImageDataset):
	datasetName = "CIFAR10"	#currently assume CIFAR-10 dataset	#expected test accuracy: ~91%
	numberOfClasses = 10
	warmupEpochs = 5 	#default: 5	#orig: 0
	learningRate = 0.001	#default: 0.001 (or 0.01)	#orig: 0.005
	momentum = 0.9     #default: 0.9	#orig: 0.0
	weightDecay  = 5e-4    #default: 5e-4	#orig: 0.0
	numberOfSamplesK = 60
	if(useAlgorithmEISANI):
		if(useCloudExecution):
			batchSize = 64//EISANICNNcontinuousVarEncodingNumBits	#default: 64
		else:
			batchSize = 1	#default: 1
		batchSize = 1	#16	#default: 64	#1024
		if(not useDefaultNumLayersParam):
			numberOfFFLayers = numberOfFFLayers*2
		numberOfLayers = numberOfConvlayers+numberOfFFLayers	#counts hidden and output layers (not input layer)
	elif(useAlgorithmRPIANN):
		batchSize = 128	#1024	#128	 #default: 128	#orig: 64
	else:
		batchSize = 128	 #default: 128	#orig: 64
		numberOfConvlayers = 6	#rest will be FF	#orig: 2	#default: 2, 4, 6
		numberOfFFLayers = 3
		numberOfLayers = numberOfConvlayers+numberOfFFLayers	#counts hidden and output layers (not input layer)
	numberInputImageChannels = 3	#default: CIFAR-10 channels
	inputImageHeight = 32	#default: CIFAR-10 image size
	inputImageWidth = 32	#default: CIFAR-10 image size
	CNNinputShape = [numberInputImageChannels, inputImageHeight, inputImageWidth]	#default: CIFAR-10 image size = 3*32*32=3072
	if(not useAlgorithmRPIANN):
		hiddenLayerSize = 1024
		CNNhiddenLayerSize = numberInputImageChannels*inputImageHeight*inputImageWidth * 4	
		if(numberOfConvlayers >= 4):
			CNNconvergeEveryEvenLayer = True	#default: True for numberOfConvlayers 4 or 6	#orig: False
			assert numberOfConvlayers%2 == 0
		else:
			CNNconvergeEveryEvenLayer = False
	imageDatasetAugment = True
	if(imageDatasetAugment):
		trainNumberOfEpochs = 100	#default: 100
	else:
		trainNumberOfEpochs = 1	#default: 10
	useLearningRateScheduler = True
	learningRateSchedulerStepsize = 10	#default: 30	#orig: 10	
	learningRateSchedulerGamma = 0.5	#default: 0.1	#orig: 0.5
	batchNormFC = False	#optional	#batch norm for fully connected layers
	dropout = False	#default: False
	dropoutProb = 0.5 	#default: 0.5	#orig: 0.3
elif(useNLPDataset):
	datasetSizeSubset = True	#default: True (required for stateTestDataset)  #if(useSequentialSANI): hiddenLayerSizeSANI is dynamically grown, and is not dependent on datasetSize (which can therefore be huge), however a) useDatasetSubset is still useful for fast training (development/debugging) and b) is required to reserve data for an eval phase 
	datasetTrainRows = 1000	#default: 100000
	datasetTestRows = int(datasetTrainRows*0.1)	#*datasetTestSplitSize
	numWorkers = 0	#default: 0	(required for stateTestDataset:datasetTestRows to be enforced) #orig = 2	#set numWorkers=1 for simplify dataset determinism during streaming (numWorkers=2 will alternate between selecting articles from wikipedia dataset shard N and shard N+1)
	batchSize = 16	#default: 16	#debug: 1
	datasetName = "wikipedia"
	datasetCfg = "20220301.en"	#not available in conda; "20231101.en", not available in huggingface; "20240501.en"
	datasetHasTestSplit = False
	trainNumberOfEpochs = 1	#default: 1	#with increased epochs can significantly increase train accuracy on train dataset (though should theoretically have no effect on test accuracy)
	
def round_up_to_power_of_2(x: float) -> int:
	if x <= 0:
		raise ValueError("x must be positive")
	return 1 << (math.ceil(math.log2(x)))

if(not datasetHasTestSplit):
	datasetTestSplitShuffle = True	#mandatory: True	#required for many datasets (without randomised class order)
	datasetTestSplitShuffleDeterministic = False	#default: False	#can be enabled to always get the same test result
	if(datasetTestSplitShuffleDeterministic):
		datasetTestSplitSeed = 1234
	else:
		datasetTestSplitSeed = None

if(useAlgorithmEISANI):
	if(useEIneurons and EIneuronsMatchComputation and useTabularDataset):
		numberOfLayers += 1
	if(trainNumberOfEpochsHigh):
		trainNumberOfEpochs = 100
	if(useTabularDataset):			
		if(not useDefaultNumLayersParam):	#useImageDataset etc does not support posthoc modification of numberOfLayers
			numberOfLayers = numberOfLayers*2
else:
	if(trainNumberOfEpochsHigh):
		trainNumberOfEpochs = trainNumberOfEpochs*4	#orig*4
	if(hiddenLayerSizeHigh):
		hiddenLayerSize = hiddenLayerSize*4
	if(numberOfLayersLow):
		numberOfLayers = 1
		
if(debugSmallBatchSize):
	batchSize = 10
if(debugSmallNetwork):
	batchSize = 2
	numberOfLayers = 4
	hiddenLayerSize = 5	
	trainNumberOfEpochs = 1
if(debugSmallDataset):
	datasetRepeat = False
		
printAccuracyRunningAverage = False
if(printAccuracyRunningAverage):
	runningAverageBatches = 10

datasetSplitNameTrain = 'train'
datasetSplitNameTest = 'test'
if(not datasetHasTestSplit):
	datasetTestSplitSize = 0.1

userName = 'user'	#default: user
tokenString = "INSERT_HUGGINGFACE_TOKEN_HERE"	#default: INSERT_HUGGINGFACE_TOKEN_HERE
import os
if(os.path.isdir('user')):
	from user.user_globalDefs import *

modelSaveNumberOfBatches = 100	#resave model after x training batches

dataFolderName = 'data'
modelFolderName = 'model'
if(relativeFolderLocations):
	dataPathName = dataFolderName
	modelPathName = modelFolderName
else:	
	dataPathName = '/media/' + userName + dataDrive + dataFolderName
	modelPathName = '/media/' + userName + workingDrive + modelFolderName

def getModelPathNameFull(modelPathNameBase, modelName):
	modelPathNameFull = modelPathNameBase + '/' + modelName + '.pt'
	return modelPathNameFull
	
modelPathNameBase = modelPathName
modelPathNameFull = getModelPathNameFull(modelPathNameBase, modelName)
	
def printCUDAmemory(tag):
	print(tag)
	
	pynvml.nvmlInit()
	h = pynvml.nvmlDeviceGetHandleByIndex(0)
	info = pynvml.nvmlDeviceGetMemoryInfo(h)
	total_memory = info.total
	memory_free = info.free
	memory_allocated = info.used
	'''
	total_memory = pt.cuda.get_device_properties(0).total_memory
	memory_reserved = pt.cuda.memory_reserved(0)
	memory_allocated = pt.cuda.memory_allocated(0)
	memory_free = memory_reserved-memory_allocated  # free inside reserved
	'''
	print("CUDA total_memory = ", total_memory)
	#print("CUDA memory_reserved = ", memory_reserved)
	print("CUDA memory_allocated = ", memory_allocated)
	print("CUDA memory_free = ", memory_free)

def printe(str):
	print(str)
	exit()


import torch as pt
useLovelyTensors = False
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	pt.set_printoptions(profile="full")
	pt.set_printoptions(sci_mode=False)

#pt.autograd.set_detect_anomaly(True)

if(useCPU):
	device = pt.device('cpu')
else:
	device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')
	
def printf(*args, filePath="log.txt", sep=" ", end="\n"):
	if(useCloudExecution):
		with open(filePath, "a") as f:
			f.write(sep.join(str(arg) for arg in args) + end)
	else:
		print(*args, sep=sep, end=end)
