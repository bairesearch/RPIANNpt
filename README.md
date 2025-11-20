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

## Projection autoencoder accuracy considerations

Enabling `inputProjectionAutoencoder` or `targetProjectionAutoencoder` adds a reconstruction objective on the projection layers before every training step. The model first runs the autoencoders and updates the projection weights, then uses those updated weights to encode the batch for the main loss.【F:RPIANNpt/RPIANNpt_RPIANNmodel.py†L319-L337】 Because the autoencoder loss is mean-squared reconstruction of the raw inputs or one-hot targets【F:RPIANNpt/RPIANNpt_RPIANNmodelAutoencoder.py†L144-L155】【F:RPIANNpt/RPIANNpt_RPIANNmodelAutoencoder.py†L218-L234】, it regularizes the projections toward identity-like mappings rather than the task-specific embeddings that maximize downstream accuracy. When projection weights would otherwise stay fixed (for example, the linear input projection is frozen when the autoencoder is disabled)【F:RPIANNpt/RPIANNpt_RPIANNmodel.py†L126-L148】, the extra reconstruction updates can introduce non-stationary inputs for the rest of the network and compete with the main optimization objective. This mismatch often leads to slightly lower classification accuracy despite the better reconstruction of the original signals.

To capture the reconstruction benefits without letting the auxiliary objective drift the projections away from the task, you can enable a short warmup and stop autoencoder updates afterward by setting `projectionAutoencoderWarmupEpochs` to a small value (for example, 3–5 epochs). This pretrains the projections for faithful reconstruction and then lets the classification loss take over, which tends to improve downstream accuracy compared with leaving the autoencoders active for every epoch.

If the autoencoders still collapse toward identity-like projections, turn them into denoising autoencoders by setting `projectionAutoencoderDenoisingStd` to a small nonzero standard deviation (for example, 0.05–0.1). During autoencoder training this injects Gaussian noise into the inputs/one-hot targets while asking the reverse projections to reconstruct the clean originals【F:RPIANNpt/RPIANNpt_RPIANN_globalDefs.py†L96-L107】【F:RPIANNpt/RPIANNpt_RPIANNmodelAutoencoder.py†L73-L78】【F:RPIANNpt/RPIANNpt_RPIANNmodelAutoencoder.py†L144-L155】【F:RPIANNpt/RPIANNpt_RPIANNmodelAutoencoder.py†L218-L234】. The projections must therefore encode denoised structure instead of copying inputs directly, which typically yields more stable embeddings for the downstream task.

To decouple the projection reconstruction objective from the main task even further, you can pretrain the autoencoders before normal RPIANN updates by setting `projectionAutoencoderPretrainEpochs` to a positive integer. This runs dedicated epochs that only optimize the projection autoencoders (using their own optimizers) before any task loss is applied, reporting the input and target reconstruction losses as it goes.【F:RPIANNpt/ANNpt_main.py†L131-L163】【F:RPIANNpt/RPIANNpt_RPIANNmodel.py†L268-L279】【F:RPIANNpt/RPIANNpt_RPIANNmodelAutoencoder.py†L80-L109】【F:RPIANNpt/RPIANNpt_RPIANN_globalDefs.py†L96-L107】 After the pretraining window, you can either keep a short warmup (`projectionAutoencoderWarmupEpochs`) or disable the autoencoders entirely, which often yields higher accuracy than interleaving reconstruction and task updates every epoch because the projections start from a well-shaped embedding space and are no longer pulled toward identity during supervised training.

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
