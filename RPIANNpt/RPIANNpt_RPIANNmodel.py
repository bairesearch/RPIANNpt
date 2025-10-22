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

class _ActionLayerBase(nn.Module):
	def __init__(self, embedding_dim, action_scale=0.25):
		super().__init__()
		if(numberOfSublayers == 1):
			self.action = nn.Sequential(
				nn.Linear(embedding_dim * 2, embedding_dim),
				nn.ReLU(),
			)
		elif(numberOfSublayers == 2):
			hidden_dim = embedding_dim * subLayerHiddenDimMultiplier
			self.action = nn.Sequential(
				nn.Linear(embedding_dim * 2, hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, embedding_dim)
			)
		else:
			printe("invalid number numberOfSublayers")
		self.action_scale = action_scale

	def forward(self, x_embed, y_hat):
		combined = pt.cat([x_embed, y_hat], dim=-1)
		# Limit the magnitude of the proposed update so the recursion remains stable
		update = pt.tanh(self.action(combined)) * self.action_scale
		return update

class RecursiveActionLayer(_ActionLayerBase):
	def __init__(self, embedding_dim, action_scale=0.25):
		super().__init__(embedding_dim, action_scale)

class NonRecursiveActionLayer(_ActionLayerBase):
	def __init__(self, embedding_dim, action_scale=0.25):
		super().__init__(embedding_dim, action_scale)

class RPIANNmodel(nn.Module):
		
	def __init__(self, config):
		super().__init__()
		self.config = config

		# Interpret the existing hidden layer size as the shared embedding dimensionality.
		self.embedding_dim = config.hiddenLayerSize
		self.recursion_steps = config.numberOfLayers

		self.using_image_projection = useCNNlayers
		self.use_recursive_layers = useRecursiveLayers
		if(self.using_image_projection):
			self.input_projection = self._build_image_projection(config)
			for param in self.input_projection.parameters():
				param.requires_grad_(False)
		else:
			self.input_projection = nn.Linear(config.inputLayerSize, self.embedding_dim, bias=False)
			self._initialise_random_linear(self.input_projection)
			self.input_projection.weight.requires_grad_(False)
		self.use_input_projection_activation = inputProjectionActivationFunction

		self.target_projection = nn.Linear(config.outputLayerSize, self.embedding_dim, bias=False)
		self._initialise_random_linear(self.target_projection)
		self.target_projection.weight.requires_grad_(False)

		self.initial_predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
		if(self.use_recursive_layers):
			self.recursive_layer = RecursiveActionLayer(self.embedding_dim)
			self.nonrecursive_layers = None
		else:
			self.recursive_layer = None
			self.nonrecursive_layers = nn.ModuleList([NonRecursiveActionLayer(self.embedding_dim) for _ in range(self.recursion_steps)])
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

	def _build_image_projection(self, config):
		input_channels, input_height, input_width = config.inputImageShape
		number_of_convs = config.numberOfConvlayers

		total_pool_downscale = 2 ** number_of_convs
		if((input_height % total_pool_downscale) != 0 or (input_width % total_pool_downscale) != 0):
			raise ValueError("Input spatial dimensions must be divisible by 2**number_of_convlayers.")

		output_height = input_height // total_pool_downscale
		output_width = input_width // total_pool_downscale
		output_spatial_size = output_height * output_width

		if(self.embedding_dim % output_spatial_size != 0):
			raise ValueError("embedding_dim must be divisible by the CNN output spatial size.")

		out_channels = self.embedding_dim // output_spatial_size
		if(out_channels <= 0):
			raise ValueError("Calculated CNN output channels must be positive.")

		use_activation = inputProjectionActivationFunction

		layers = []
		in_channels = input_channels
		for idx in range(number_of_convs):
			conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
			self._initialise_random_conv(conv)
			conv.weight.requires_grad_(False)
			layers.append(conv)
			layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
			if(use_activation):
				layers.append(nn.ReLU())
			in_channels = out_channels

		layers.append(nn.Flatten(start_dim=1))

		projection = nn.Sequential(*layers)

		for module in projection.modules():
			if isinstance(module, nn.Conv2d):
				module.weight.requires_grad_(False)

		return projection

	def _compute_total_loss(self, y_hat, target_embedding, y):
		logits = self.project_to_classes(y_hat)
		classification_loss = self.lossFunctionFinal(logits, y)
		embedding_alignment_loss = F.mse_loss(y_hat, target_embedding)
		if(useClassificationLayerLoss):
			total_loss = classification_loss + self.embedding_loss_weight * embedding_alignment_loss
		else:
			total_loss = embedding_alignment_loss
		return total_loss, logits, classification_loss, embedding_alignment_loss

	def encode_input(self, x):
		if(self.using_image_projection):
			return self.input_projection(x)
		if(useImageDataset):
			batch_size = x.shape[0]
			x = x.reshape(batch_size, -1)
		projected = self.input_projection(x)
		if(self.use_input_projection_activation):
			projected = F.relu(projected)
		return projected

	def encode_targets(self, y):
		y_one_hot = F.one_hot(y, num_classes=self.config.outputLayerSize).float()
		return self.target_projection(y_one_hot)

	def initialise_prediction(self, x_embed):
		return pt.tanh(self.initial_predictor(x_embed))

	def iterate_prediction(self, x_embed, y_hat):
		if(self.use_recursive_layers):
			for _ in range(self.recursion_steps):
				y_hat = y_hat + self.recursive_layer(x_embed, y_hat)
		else:
			for layer in self.nonrecursive_layers:
				y_hat = y_hat + layer(x_embed, y_hat)
		return y_hat

	def _local_step(self, forward_fn, target_embedding, y, optim, index):
		opt = self._resolve_local_optimizer(optim, index)
		if opt is not None:
			y_hat = forward_fn()
			total_loss, _, _, _ = self._compute_total_loss(y_hat, target_embedding, y)
			opt.zero_grad()
			total_loss.backward()
			opt.step()
		with pt.no_grad():
			y_hat = forward_fn()
		return y_hat.detach()

	def _train_recursive_locally(self, x_embed, target_embedding, y, optim):
		def initial_forward():
			return pt.tanh(self.initial_predictor(x_embed))

		y_hat_state = self._local_step(initial_forward, target_embedding, y, optim, index=0)

		if(self.use_recursive_layers):
			for step in range(self.recursion_steps):
				base_state = y_hat_state
				def iterate_forward(base=base_state):
					return base + self.recursive_layer(x_embed, base)
				y_hat_state = self._local_step(iterate_forward, target_embedding, y, optim, index=step+1)
		else:
			for step, layer in enumerate(self.nonrecursive_layers):
				base_state = y_hat_state
				def iterate_forward(base=base_state, layer_ref=layer):
					return base + layer_ref(x_embed, base)
				y_hat_state = self._local_step(iterate_forward, target_embedding, y, optim, index=step+1)

		return y_hat_state

	def project_to_classes(self, y_hat):
		# Using the frozen target projection weights as a random classifier head.
		return pt.matmul(y_hat, self.target_projection.weight)
			
	def forward(self, trainOrTest, x, y, optim, layer=None):
		x_embed = self.encode_input(x)
		target_embedding = self.encode_targets(y)

		if(trainOrTest and trainLocal):
			x_embed = x_embed.detach()
			target_embedding = target_embedding.detach()
			y_hat = self._train_recursive_locally(x_embed, target_embedding, y, optim)
			with pt.no_grad():
				total_loss, logits, classification_loss, embedding_alignment_loss = self._compute_total_loss(y_hat, target_embedding, y)
			loss = total_loss.detach()
			accuracy = self.accuracyFunction(logits, y)
			self.last_y_hat = y_hat
			self.last_logits = logits.detach()
			return loss, accuracy
		else:
			if(trainOrTest):
				x_embed = x_embed.requires_grad_()
			target_embedding = target_embedding.detach()
			y_hat_initial = self.initialise_prediction(x_embed)
			y_hat = self.iterate_prediction(x_embed, y_hat_initial)
			total_loss, logits, _, _ = self._compute_total_loss(y_hat, target_embedding, y)
			accuracy = self.accuracyFunction(logits, y)
			self.last_y_hat = y_hat.detach()
			self.last_logits = logits.detach()
			return total_loss, accuracy

	def _resolve_local_optimizer(self, optim, index=0):
		if(optim is None):
			return None
		if(isinstance(optim, list)):
			if(len(optim) == 0):
				return None
			selected = optim[min(index, len(optim)-1)]
			if(isinstance(selected, list)):
				if(len(selected) == 0):
					return None
				return selected[0]
			return selected
		return optim
