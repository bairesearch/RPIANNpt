"""RPIANNpt_RPIANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
RPIANNpt Recursive Prediction Improvement artificial neural network model

"""

import math

import torch as pt
from torch import nn
import torch.nn.functional as F
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
if(useRPICNN):
	from RPIANNpt_RPIANNmodelLayerCNN import *
else:
	from RPIANNpt_RPIANNmodelLayerMLP import *

class RPIANNconfig():
	def __init__(self, batchSize, numberOfLayers, numberOfConvlayers, hiddenLayerSize, CNNhiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples, inputImageShape=None):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.numberOfConvlayers = numberOfConvlayers
		self.hiddenLayerSize = hiddenLayerSize
		self.CNNhiddenLayerSize = CNNhiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.datasetSize = datasetSize		
		self.numberOfClassSamples = numberOfClassSamples
		self.inputImageShape = inputImageShape

class RPIANNmodel(nn.Module):
		
	def __init__(self, config):
		super().__init__()
		self.config = config

		# Interpret the existing hidden layer size as the shared embedding dimensionality.
		self.embedding_dim = config.hiddenLayerSize
		self.recursion_steps = config.numberOfLayers

		self.using_image_projection = useCNNlayers
		self.using_rpi_cnn = useRPICNN
		if(useImageDataset):	
			projection_stride = CNNprojectionStride
		self._x_feature_shape = None
		self._y_feature_shape = None
		self._x_feature_size = None
		self._y_feature_size = self.embedding_dim
		self.use_recursive_layers = useRecursiveLayers
		self.use_hidden_activation = hiddenActivationFunction

		if(self.using_image_projection):
			if(self.using_rpi_cnn):
				self.input_projection, feature_shape = self._build_image_projection(config, stride=projection_stride, trainable=False)
				self._x_feature_shape = feature_shape
				self._y_feature_shape = self._determine_cnn_feature_shape(config, projection_stride)
				self._x_feature_size = self._feature_size(self._x_feature_shape)
				self._y_feature_size = self._feature_size(self._y_feature_shape)
			else:
				self.input_projection, feature_shape = self._build_image_projection(config, stride=projection_stride, trainable=False)
				self._x_feature_shape = feature_shape
				self._y_feature_shape = feature_shape
				self._x_feature_size = self._feature_size(feature_shape)
				self._y_feature_size = self._x_feature_size
		else:
			if(useImageDataset):	
				self.input_projection = nn.Sequential(nn.Flatten(start_dim=1))
				if(self.using_rpi_cnn):
					self._x_feature_shape = tuple(config.inputImageShape)
					self._y_feature_shape = self._determine_cnn_feature_shape(config, projection_stride)
					self._x_feature_size = self._feature_size(self._x_feature_shape)
					self._y_feature_size = self._feature_size(self._y_feature_shape)
			else:
				self.input_projection = nn.Linear(config.inputLayerSize, self.embedding_dim, bias=False)
				self._initialise_random_linear(self.input_projection)
				self.input_projection.weight.requires_grad_(False)

		self.target_projection = nn.Linear(config.outputLayerSize, self.embedding_dim, bias=False)
		self._initialise_random_linear(self.target_projection)
		self.target_projection.weight.requires_grad_(False)

		if(inputProjectionActivationFunction):
			self.input_projection_activation = nn.ReLU()
		else:
			self.input_projection_activation = nn.Identity()
		if(inputProjectionActivationFunctionTanh):
			self.input_projection_activation_tanh = nn.Tanh()
		else:
			self.input_projection_activation_tanh = nn.Identity()

		if(targetProjectionActivationFunction):
			self.target_projection_activation = nn.ReLU()
		else:
			self.target_projection_activation = nn.Identity()
		if(targetProjectionActivationFunctionTanh):
			self.target_projection_activation_tanh = nn.Tanh()
		else:
			self.target_projection_activation_tanh = nn.Identity()

		self.initial_predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
		if(self.use_recursive_layers):
			if(self.using_rpi_cnn):
				self.recursive_layer = RecursiveCNNActionLayer(self._y_feature_shape, numberOfSublayers, layerScale, use_activation=self.use_hidden_activation, x_feature_shape=self._x_feature_shape)
			else:
				self.recursive_layer = RecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation)
			self.nonrecursive_layers = None
		else:
			self.recursive_layer = None
			if(self.using_rpi_cnn):
				self.nonrecursive_layers = nn.ModuleList([NonRecursiveCNNActionLayer(self._y_feature_shape, numberOfSublayers, layerScale, use_activation=self.use_hidden_activation, x_feature_shape=self._x_feature_shape) for _ in range(self.recursion_steps)])
			else:
				self.nonrecursive_layers = nn.ModuleList([NonRecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation) for _ in range(self.recursion_steps)])
		if(useClassificationLayerLoss):
			self.embedding_loss_weight = 0.1

		self.lossFunctionFinal = nn.CrossEntropyLoss()
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		self.last_y_hat = None
		self.last_logits = None

	def _initialise_random_linear(self, module):
		std = 1.0 / math.sqrt(module.out_features)
		nn.init.normal_(module.weight, mean=0.0, std=std)

	def _initialise_random_conv(self, module):
		kernel_height, kernel_width = module.kernel_size
		fan_out = module.out_channels * kernel_height * kernel_width
		std = 1.0 / math.sqrt(fan_out)
		nn.init.normal_(module.weight, mean=0.0, std=std)

	def _feature_size(self, feature_shape):
		channels, height, width = feature_shape
		return channels * height * width

	def _determine_cnn_feature_shape(self, config, stride):
		if(config.inputImageShape is None):
			raise ValueError("Image projection requires an input image shape.")
		number_of_convs = config.numberOfConvlayers or 1
		if(number_of_convs <= 0):
			raise ValueError("numberOfConvlayers must be a positive integer.")
		input_channels, input_height, input_width = config.inputImageShape
		if(stride == 1):
			output_height = input_height
			output_width = input_width
		elif(stride == 2):
			total_pool_downscale = stride ** number_of_convs
			if((input_height % total_pool_downscale) != 0 or (input_width % total_pool_downscale) != 0):
				raise ValueError("Input spatial dimensions must be divisible by stride**number_of_convlayers.")
			output_height = input_height // total_pool_downscale
			output_width = input_width // total_pool_downscale
		else:
			raise ValueError("Image projection stride must be 1 or 2.")
		output_spatial_size = output_height * output_width
		if(output_spatial_size <= 0):
			raise ValueError("Calculated CNN output spatial size must be positive.")
		if(self.embedding_dim % output_spatial_size != 0):
			raise ValueError("embedding_dim must be divisible by the CNN output spatial size.")
		out_channels = self.embedding_dim // output_spatial_size
		if(out_channels <= 0):
			raise ValueError("Calculated CNN output channels must be positive.")
		return (out_channels, output_height, output_width)

	def _build_image_projection(self, config, stride=2, trainable=False):
		if(config.inputImageShape is None):
			raise ValueError("Image projection requires a defined input image shape.")
		if(stride not in (1, 2)):
			raise ValueError("Image projection stride must be 1 or 2.")
		number_of_convs = config.numberOfConvlayers or 1
		if(number_of_convs <= 0):
			raise ValueError("numberOfConvlayers must be a positive integer.")

		input_channels, _, _ = config.inputImageShape
		feature_shape = self._determine_cnn_feature_shape(config, stride)
		out_channels, _, _ = feature_shape

		use_activation = imageProjectionActivationFunction

		layers = []
		in_channels = input_channels
		for _ in range(number_of_convs):
			conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
			self._initialise_random_conv(conv)
			conv.weight.requires_grad_(trainable)
			layers.append(conv)
			if(stride == 1):
				layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
			else:
				layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
			if(use_activation):
				layers.append(nn.ReLU())
			in_channels = out_channels

		layers.append(nn.Flatten(start_dim=1))

		projection = nn.Sequential(*layers)
		return projection, feature_shape
			
	def forward(self, trainOrTest, x, y, optim, layer=None):
		x_embed = self.encode_input(x)
		target_embedding = self.encode_targets(y)

		if(trainOrTest and trainLocal):
			x_embed = x_embed.detach()
			target_embedding = target_embedding.detach()
			y_hat = self._train_recursive_locally(x_embed, target_embedding, y, optim)
			with pt.no_grad():
				total_loss, logits, classification_loss, embedding_alignment_loss, activated_y_hat = self._compute_total_loss(y_hat, target_embedding, y)
			loss = total_loss.detach()
			accuracy = self.accuracyFunction(logits, y)
			self.last_y_hat = activated_y_hat.detach()
			self.last_logits = logits.detach()
			return loss, accuracy
		else:
			if(trainOrTest):
				x_embed = x_embed.requires_grad_()
			target_embedding = target_embedding.detach()
			y_hat = self.iterate_prediction(x_embed)
			total_loss, logits, _, _, activated_y_hat = self._compute_total_loss(y_hat, target_embedding, y)
			accuracy = self.accuracyFunction(logits, y)
			self.last_y_hat = activated_y_hat.detach()
			self.last_logits = logits.detach()
			return total_loss, accuracy

	def _train_recursive_locally(self, x_embed, target_embedding, y, optim):
		if(initialiseYhatZero or x_embed.shape[1] != self._y_feature_size):
			y_hat_state = self._zero_y_hat_like(x_embed)
		else:
			y_hat_state = x_embed
		if(self.use_recursive_layers):
			for step in range(self.recursion_steps):
				update = self._local_step(y_hat_state, x_embed, target_embedding, y, optim, index=step)
				y_hat_state = self._applyResidual(y_hat_state, update)
		else:
			for step, layer in enumerate(self.nonrecursive_layers):
				update = self._local_step(y_hat_state, x_embed, target_embedding, y, optim, index=step, layer_ref=layer)
				y_hat_state = self._applyResidual(y_hat_state, update)

		return update

	def _local_step(self, base, x_embed, target_embedding, y, optim, index, layer_ref=None):
		opt = self._resolve_local_optimizer(optim, index)
		y_hat = self._local_step_forward(base, x_embed, layer_ref)
		total_loss, *_ = self._compute_total_loss(y_hat, target_embedding, y)
		opt.zero_grad()
		total_loss.backward()
		opt.step()
		y_hat = y_hat.detach()
		return y_hat

	def _resolve_local_optimizer(self, optim, index):	
		return optim[index]
	
	def _local_step_forward(self, base, x_embed, layer_ref=None):
		if(self.use_recursive_layers):
			y_hat = self._iterate_forward_recursive(base, x_embed)
		else:
			y_hat = self._iterate_forward_nonrecursive(base, x_embed, layer_ref)
		return y_hat
	
	def _iterate_forward_recursive(self, base, x_embed):
		out = self.recursive_layer(base, x_embed)
		return out
	
	def _iterate_forward_nonrecursive(self, base, x_embed, layer_ref):
		out = layer_ref(base, x_embed)
		return out
	
	def iterate_prediction(self, x_embed):
		if(initialiseYhatZero or x_embed.shape[1] != self._y_feature_size):
			y_hat = self._zero_y_hat_like(x_embed)
		else:
			y_hat = x_embed
		if(self.use_recursive_layers):
			for _ in range(self.recursion_steps):
				update = self.recursive_layer(y_hat, x_embed)
				y_hat = self._applyResidual(y_hat, update)
		else:
			for layer in self.nonrecursive_layers:
				update = layer(y_hat, x_embed)
				y_hat = self._applyResidual(y_hat, update)
		return y_hat
	
	def _applyResidual(self, orig, update):
		if(layersFeedResidualInput):
			return orig + update
		else:
			return update
			
	def _compute_total_loss(self, y_hat, target_embedding, y):
		logits = self.project_to_classes(y_hat)
		classification_loss = self.lossFunctionFinal(logits, y)
		embedding_alignment_loss = F.mse_loss(y_hat, target_embedding)
		if(useClassificationLayerLoss):
			total_loss = classification_loss + self.embedding_loss_weight * embedding_alignment_loss
		else:
			total_loss = embedding_alignment_loss
		return total_loss, logits, classification_loss, embedding_alignment_loss, y_hat

	def project_to_classes(self, y_hat):
		# Using the frozen target projection weights as a random classifier head.
		return pt.matmul(y_hat, self.target_projection.weight)

	def _zero_y_hat_like(self, reference_tensor):
		return reference_tensor.new_zeros((reference_tensor.shape[0], self._y_feature_size))
	
	def encode_input(self, x):
		projected = self.encode_inputs(x)
		return projected

	def encode_targets(self, y):
		y_one_hot = F.one_hot(y, num_classes=self.config.outputLayerSize).float()
		target_embedding = self.target_projection(y_one_hot)
		target_embedding = self.target_projection_activation(target_embedding)
		target_embedding = self.target_projection_activation_tanh(target_embedding)
		return target_embedding
	
	def encode_inputs(self, x):
		input_embedding = self.input_projection(x)
		input_embedding = self.input_projection_activation(input_embedding)
		input_embedding = self.input_projection_activation_tanh(input_embedding)
		return input_embedding
