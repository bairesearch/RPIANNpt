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

class SeparateStreamsFirstSublayer(nn.Module):
	def __init__(self, embedding_dim, hidden_dim):
		super().__init__()
		if(hidden_dim < 2):
			raise ValueError("hidden_dim must be at least 2 to split x_embed and y_hat streams separately.")
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.x_out_features = math.ceil(hidden_dim / 2.0)
		self.y_out_features = hidden_dim - self.x_out_features
		if(self.y_out_features == 0):
			raise ValueError("hidden_dim too small to allocate features to both x_embed and y_hat streams.")
		self.x_linear = nn.Linear(embedding_dim, self.x_out_features)
		self.y_linear = nn.Linear(embedding_dim, self.y_out_features)

	def forward(self, combined):
		expected_features = self.embedding_dim * 2
		if(combined.shape[-1] != expected_features):
			raise ValueError(f"SeparateStreamsFirstSublayer expected input feature dimension {expected_features}, received {combined.shape[-1]}.")
		x_embed = combined[..., :self.embedding_dim]
		y_hat = combined[..., self.embedding_dim:]
		x_proj = self.x_linear(x_embed)
		y_proj = self.y_linear(y_hat)
		return pt.cat([x_proj, y_proj], dim=-1)

class _ActionLayerBase(nn.Module):
	def __init__(self, embedding_dim, action_scale, use_activation=True):
		super().__init__()
		if(use_activation):
			activation_module = nn.ReLU()
		else:
			activation_module = nn.Identity()
		if(numberOfSublayers == 1):
			hidden_dim = embedding_dim
			if(layersFeedConcatInput):
				first_layer = nn.Linear(embedding_dim * 2, hidden_dim)
			else:
				first_layer = nn.Linear(embedding_dim, hidden_dim)
			self.action = nn.Sequential(
				first_layer,
				activation_module,
			)
		elif(numberOfSublayers == 2):
			hidden_dim = embedding_dim * subLayerHiddenDimMultiplier
			if(layersFeedConcatInput):
				if(subLayerFirstMixXembedYhatStreamsSeparately):
					first_layer = SeparateStreamsFirstSublayer(embedding_dim, hidden_dim)
				else:
					first_layer = nn.Linear(embedding_dim * 2, hidden_dim)
			else:
				first_layer = nn.Linear(embedding_dim, hidden_dim)
			if(subLayerFirstNotTrained):
				self._initialise_random_linear_layer(first_layer)
				for param in first_layer.parameters():
					param.requires_grad_(False)
			second_layer = nn.Linear(hidden_dim, embedding_dim)
			self.action = nn.Sequential(
				first_layer,
				nn.ReLU(),
				second_layer,
				activation_module
			)
		else:
			printe("invalid number numberOfSublayers")
		self.action_scale = action_scale
	
	def _initialise_random_linear_layer(self, layer):
		if(isinstance(layer, SeparateStreamsFirstSublayer)):
			self._initialise_dense_linear(layer.x_linear)
			self._initialise_dense_linear(layer.y_linear)
			if subLayerFirstSparse:
				self._apply_sparse_mask(layer.x_linear.weight, subLayerFirstSparsityLevel)
				self._apply_sparse_mask(layer.y_linear.weight, subLayerFirstSparsityLevel)
		else:
			self._initialise_dense_linear(layer)
			if subLayerFirstSparse:
				self._apply_sparse_mask(layer.weight, subLayerFirstSparsityLevel)

	def _initialise_dense_linear(self, linear_module):
		std = 1.0 / math.sqrt(linear_module.out_features)
		nn.init.normal_(linear_module.weight, mean=0.0, std=std)
		if(linear_module.bias is not None):
			nn.init.zeros_(linear_module.bias)

	def _apply_sparse_mask(self, weight, sparsity_level):
		if(sparsity_level == 0.0):
			return
		if(sparsity_level < 0.0 or sparsity_level >= 1.0):
			raise ValueError("subLayerFirstSparsityLevel must be within [0.0, 1.0) to retain connections.")
		keep_fraction = 1.0 - sparsity_level
		in_features = weight.shape[1]
		target_connections = max(1, int(math.ceil(keep_fraction * in_features)))
		target_connections = min(target_connections, in_features)
		if(target_connections == in_features):
			return
		with pt.no_grad():
			mask = weight.new_zeros(weight.shape)
			for row in range(weight.shape[0]):
				selected_indices = pt.randperm(in_features, device=weight.device)[:target_connections]
				mask[row, selected_indices] = 1.0
			weight.mul_(mask)

	def forward(self, y_hat, x_embed):
		if(layersFeedConcatInput):
			combined = pt.cat([x_embed, y_hat], dim=-1)
		else:
			combined = y_hat
		raw_update = self.action(combined)
		if(hiddenActivationFunctionTanh):
			raw_update = pt.tanh(raw_update)
		update = raw_update * self.action_scale	# Limit the magnitude of the proposed update so the recursion remains stable
		return update

class RecursiveActionLayer(_ActionLayerBase):
	def __init__(self, embedding_dim, action_scale, use_activation=True):
		super().__init__(embedding_dim, action_scale, use_activation)

class NonRecursiveActionLayer(_ActionLayerBase):
	def __init__(self, embedding_dim, action_scale, use_activation=True):
		super().__init__(embedding_dim, action_scale, use_activation)

class _CNNActionLayerBase(nn.Module):
	def __init__(self, y_feature_shape, number_of_conv_layers, action_scale, use_activation=True, x_feature_shape=None):
		super().__init__()
		if(y_feature_shape is None):
			raise ValueError("RPICNN requires target feature shape for y_hat.")
		if(number_of_conv_layers is None or number_of_conv_layers <= 0):
			raise ValueError("numberOfConvlayers must be a positive integer when useRPICNN is enabled.")
		self.y_channels, self.feature_height, self.feature_width = y_feature_shape
		if(x_feature_shape is None):
			self.x_channels = self.y_channels
		else:
			self.x_channels, x_height, x_width = x_feature_shape
			if(x_height != self.feature_height or x_width != self.feature_width):
				raise ValueError("RPICNN requires x_embed and y_hat to share spatial dimensions.")
		self.y_feature_size = self.y_channels * self.feature_height * self.feature_width
		self.x_feature_size = self.x_channels * self.feature_height * self.feature_width
		self.number_of_conv_layers = number_of_conv_layers
		self.action_scale = action_scale
		self.use_activation = use_activation

		if(layersFeedConcatInput):
			conv_in_channels = self.x_channels + self.y_channels
		else:
			conv_in_channels = self.y_channels
		conv_out_channels = self.y_channels

		self.conv_layers = nn.ModuleList()
		self.pool_layers = nn.ModuleList()
		for _ in range(self.number_of_conv_layers):
			conv = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=3, stride=1, padding=1, bias=True)
			self._initialise_dense_conv(conv)
			self.conv_layers.append(conv)
			pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
			self.pool_layers.append(pool)
		if(self.use_activation):
			self.activation = nn.ReLU()
		else:
			self.activation = nn.Identity()

	def _initialise_dense_conv(self, conv_module):
		kernel_height, kernel_width = conv_module.kernel_size
		fan_out = conv_module.out_channels * kernel_height * kernel_width
		std = 1.0 / math.sqrt(fan_out)
		nn.init.normal_(conv_module.weight, mean=0.0, std=std)
		if(conv_module.bias is not None):
			nn.init.zeros_(conv_module.bias)

	def _reshape_flat_to_feature(self, tensor, expected_size, channels, tensor_name):
		if(tensor.dim() != 2 or tensor.shape[1] != expected_size):
			raise ValueError(f"RPICNN layer expected flattened tensor '{tensor_name}' with feature dimension {expected_size}, received shape {tensor.shape}.")
		return tensor.reshape(tensor.shape[0], channels, self.feature_height, self.feature_width)

	def _reshape_feature_to_flat(self, tensor):
		return tensor.reshape(tensor.shape[0], -1)

	def forward(self, y_hat, x_embed):
		y_hat_feature = self._reshape_flat_to_feature(y_hat, self.y_feature_size, self.y_channels, "y_hat")
		if(layersFeedConcatInput):
			x_embed_feature = self._reshape_flat_to_feature(x_embed, self.x_feature_size, self.x_channels, "x_embed")
		else:
			x_embed_feature = None
		current = y_hat_feature
		for conv, pool in zip(self.conv_layers, self.pool_layers):
			if(layersFeedConcatInput):
				combined = pt.cat([x_embed_feature, current], dim=1)
			else:
				combined = current
			out = conv(combined)
			out = self.activation(out)
			current = pool(out)
		if(hiddenActivationFunctionTanh):
			current = pt.tanh(current)
		current = current * self.action_scale
		return self._reshape_feature_to_flat(current)

class RecursiveCNNActionLayer(_CNNActionLayerBase):
	def __init__(self, y_feature_shape, number_of_conv_layers, action_scale, use_activation=True, x_feature_shape=None):
		super().__init__(y_feature_shape, number_of_conv_layers, action_scale, use_activation, x_feature_shape=x_feature_shape)

class NonRecursiveCNNActionLayer(_CNNActionLayerBase):
	def __init__(self, y_feature_shape, number_of_conv_layers, action_scale, use_activation=True, x_feature_shape=None):
		super().__init__(y_feature_shape, number_of_conv_layers, action_scale, use_activation, x_feature_shape=x_feature_shape)

class RPIANNmodel(nn.Module):
		
	def __init__(self, config):
		super().__init__()
		self.config = config

		# Interpret the existing hidden layer size as the shared embedding dimensionality.
		self.embedding_dim = config.hiddenLayerSize
		self.recursion_steps = config.numberOfLayers

		self.using_image_projection = useCNNlayers
		self.using_rpi_cnn = useRPICNN
		self._x_feature_shape = None
		self._y_feature_shape = None
		self._x_feature_size = None
		self._y_feature_size = self.embedding_dim
		self.use_recursive_layers = useRecursiveLayers
		self.use_hidden_activation = hiddenActivationFunction

		if(self.using_image_projection):
			projection_stride = CNNprojectionStride
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
				self.recursive_layer = RecursiveCNNActionLayer(self._y_feature_shape, config.numberOfConvlayers, layerScale, use_activation=self.use_hidden_activation, x_feature_shape=self._x_feature_shape)
			else:
				self.recursive_layer = RecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation)
			self.nonrecursive_layers = None
		else:
			self.recursive_layer = None
			if(self.using_rpi_cnn):
				self.nonrecursive_layers = nn.ModuleList([NonRecursiveCNNActionLayer(self._y_feature_shape, config.numberOfConvlayers, layerScale, use_activation=self.use_hidden_activation, x_feature_shape=self._x_feature_shape) for _ in range(self.recursion_steps)])
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
