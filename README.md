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

![RecursivePredictionImprovementImplementation1a.png](https://github.com/bairesearch/RPIANNpt/blob/main/graph/RecursivePredictionImprovementImplementation1a.png?raw=true)

## RPIANN vs Backprop performance (classification layer loss)

![RecursivePredictionImprovementVsBackpropTestAccuracy-SMALL.png](https://github.com/bairesearch/RPIANNpt/blob/main/graph/RecursivePredictionImprovementVsBackpropTestAccuracy-SMALL.png?raw=true)

RPIANNpt tests conducted with settings;
```
useRecursiveLayers = True
useClassificationLayerLoss = True (uses backprop calculations from target layer loss)
numberOfLayers = (lay+1)
numberOfSublayers = 1 (sublayers per layer)
hiddenLayerSizeHigh = True (units*4)
inputProjectionActivationFunction = False
useImageDataset = False
```

## RPIANN vs Backprop performance (embedding layer loss)

![RPIANNtestAccuracyImageDataset-SMALL.png](https://github.com/bairesearch/RPIANNpt/blob/main/graph/RPIANNtestAccuracyImageDataset-SMALL.png?raw=true)

RPIANNpt tests conducted with settings;
```
useRecursiveLayers = False
useClassificationLayerLoss = False (uses embedding layer loss calculated by reverse projection from target layer)
numberOfLayers = 1
numberOfSublayers = 1 (sublayers per layer)
hiddenLayerSizeHigh = True
inputProjectionActivationFunction = True
useImageDataset = True
useCNNlayers = True
numberOfConvlayers = 1
```


### Related Work

* Breakaway Network (https://github.com/bairesearch/AEANNpt)
* Recursive Transformer (https://github.com/bairesearch/TSBNLPpt)
* Large Untrained Network (https://github.com/bairesearch/LUANNpt)
* Local Connectome (https://github.com/bairesearch/LocalConnectome)
* Li, Q., Teh, Y. W., & Pascanu, R. (2025). NoProp: Training Neural Networks without Back-propagation or Forward-propagation. arXiv preprint arXiv:2503.24322.
* Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., ... & Yadkori, Y. A. (2025). Hierarchical Reasoning Model. arXiv preprint arXiv:2506.21734.
* Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks. arXiv preprint arXiv:2510.04871.
* Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8), 2554-2558.
