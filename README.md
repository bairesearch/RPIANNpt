# RPIANNpt

### Author

Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

### Description

Recursive Prediction Improvement artificial neural network for PyTorch - experimental 

### License

MIT License

### Installation
```
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics
pip install torchvision
pip install torchsummary
```

### Execution
```
source activate pytorchsenv
python ANNpt_main.py
```

## Recursive Prediction Improvement

![RPIANNImplementation1a.png](https://github.com/bairesearch/RPIANNpt/blob/main/graph/RPIANNImplementation1a.png?raw=true)

## RPIANN vs ANN performance Tabular Dataset (classification layer loss)

![RPIANNVsBackpropTestAccuracy-SMALL.png](https://github.com/bairesearch/RPIANNpt/blob/main/graph/RPIANNVsBackpropTestAccuracy-SMALL.png?raw=true)

RPIANN tests conducted with settings;
```
useTabularDataset = True
useRecursiveLayers = True
trainLocal = True
numberOfLayers = (lay+1)
useClassificationLayerLoss = True (uses backprop calculations from target layer loss)
trainClassificationLayer = False

numberOfSublayers = 1 (sublayers per layer)
subLayerFirstNotTrained = True (if more than one sublayer, first is not trained)
hiddenLayerSizeHigh = True (units*4)
inputProjectionActivationFunction = False
inputProjectionActivationFunctionTanh = False
targetProjectionActivationFunction = True
targetProjectionActivationFunctionTanh = False
layersFeedConcatInput = True
layersFeedResidualInput = True
initialiseZzero = False
batchSize = 64
```

## RPIANN performance Image Dataset (embedding layer loss)

![RPIANNtestAccuracyImageDataset-SMALL.png](https://github.com/bairesearch/RPIANNpt/blob/main/graph/RPIANNtestAccuracyImageDataset-SMALL.png?raw=true)

RPIANN tests conducted with settings;
```
useImageDataset = True
useRecursiveLayers = True or False
trainLocal = True or False
numberOfLayers = 9 or 1
useClassificationLayerLoss = False (uses embedding layer loss calculated by reverse projection from target layer)
trainClassificationLayer = False

numberOfSublayers = 1 (sublayers per layer)
subLayerFirstNotTrained = True (if more than one sublayer, first is not trained)
hiddenLayerSizeHigh = True
inputProjectionActivationFunction = True
inputProjectionActivationFunctionTanh = True
targetProjectionActivationFunction = True
targetProjectionActivationFunctionTanh = True
layersFeedConcatInput = True
layersFeedResidualInput = False
initialiseZzero = True
useCNNinputProjection = True
CNNprojectionNumlayers = 1
batchSize = 1024
```

## RPIANN performance Image Dataset (classification layer loss+train)

![RPIANNtestAccuracyImageDatasetTrainClassificationLayer-SMALL.png](https://github.com/bairesearch/RPIANNpt/blob/main/graph/RPIANNtestAccuracyImageDatasetTrainClassificationLayer-SMALL.png?raw=true)

RPIANN tests conducted with settings;
```
useImageDataset = True
useRecursiveLayers = True or False
trainLocal = True or False
numberOfLayers = 9 or 1
useClassificationLayerLoss = True (uses backprop calculations from target layer loss)
trainClassificationLayer = True (also trains classification layer)

numberOfSublayers = 1 (sublayers per layer)
subLayerFirstNotTrained = True (if more than one sublayer, first is not trained)
hiddenLayerSizeHigh = True
inputProjectionActivationFunction = True
inputProjectionActivationFunctionTanh = True
targetProjectionActivationFunction = True
targetProjectionActivationFunctionTanh = True
layersFeedConcatInput = True
layersFeedResidualInput = False
initialiseZzero = True
useCNNinputProjection = True
CNNprojectionNumlayers = 1
batchSize = 1024
```

### Related Work

* Breakaway Network (https://github.com/bairesearch/AEANNpt)
* Recursive Transformer (https://github.com/bairesearch/TSBNLPpt)
* Large Untrained Network (https://github.com/bairesearch/LUANNpt)
* Local Connectome (https://github.com/bairesearch/LocalConnectome)
* Excitatory Inhibitory Summation Activated Neuronal Input network (https://github.com/bairesearch/EISANIpt)
* Li, Q., Teh, Y. W., & Pascanu, R. (2025). NoProp: Training Neural Networks without Back-propagation or Forward-propagation. arXiv preprint arXiv:2503.24322.
* Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., ... & Yadkori, Y. A. (2025). Hierarchical Reasoning Model. arXiv preprint arXiv:2506.21734.
* Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks. arXiv preprint arXiv:2510.04871.
* Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8), 2554-2558.
