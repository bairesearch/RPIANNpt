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

## Recursive Prediction Improvement vs Backprop performance

![RecursivePredictionImprovementVsBackpropTestAccuracy-SMALL.png](https://github.com/bairesearch/RPIANNpt/blob/main/graph/RecursivePredictionImprovementVsBackpropTestAccuracy-SMALL.png?raw=true)

RPIANNpt tests conducted with settings;
```
recursion_steps = "numberOfLayers" (lay+1)
recursiveActionLayers = 1 (MLP sublayers per recursive step)
hiddenLayerSizeHigh = True (units*4)
```

### Related Work

* Breakaway Network (https://github.com/bairesearch/AEANNpt)
* Recursive Transformer (https://github.com/bairesearch/TSBNLPpt)
* Li, Q., Teh, Y. W., & Pascanu, R. (2025). NoProp: Training Neural Networks without Back-propagation or Forward-propagation. arXiv preprint arXiv:2503.24322.
* Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., ... & Yadkori, Y. A. (2025). Hierarchical Reasoning Model. arXiv preprint arXiv:2506.21734. (outer refinement loop)
* Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks. arXiv preprint arXiv:2510.04871. (outer refinement loop)
