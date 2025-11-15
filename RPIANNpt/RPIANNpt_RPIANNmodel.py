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
from RPIANNpt_RPIANNmodelLayerMLP import NonRecursiveActionLayer, RecursiveActionLayer
if(useRPICNN):
	from RPIANNpt_RPIANNmodelLayerCNN import NonRecursiveCNNActionLayer, RecursiveCNNActionLayer

class RPIANNconfig():
	def __init__(self, batchSize, numberOfLayers, numberOfConvlayers, hiddenLayerSize, CNNhiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples, inputImageShape=None, class_exemplar_images=None):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.numberOfConvlayers = numberOfConvlayers
		self.numberOfFFLayers = numberOfLayers - numberOfConvlayers
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
		self.class_exemplar_images = class_exemplar_images

class RPIANNmodel(nn.Module):
		
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.train_classification_layer = bool(useClassificationLayerLoss and trainClassificationLayer)
		if(trainClassificationLayer and not useClassificationLayerLoss):
			print("Warning: trainClassificationLayer=True has no effect when useClassificationLayerLoss=False.")

		# Interpret the existing hidden layer size as the shared embedding dimensionality.
		self.embedding_dim = config.hiddenLayerSize
		self.recursion_steps = config.numberOfLayers
		self.use_recursive_layers = useRecursiveLayers
		self.use_hidden_activation = hiddenActivationFunction
		self.target_projection_unique_per_layer = targetProjectionUniquePerLayer
	
		self.use_target_exemplar_projection = targetProjectionExemplarImage
		self.using_input_image_projection = useCNNinputProjection
		self.using_target_image_projection = useCNNtargetProjection
		self.using_rpi_cnn = useRPICNN
		if(useImageDataset):	
			projection_stride = CNNprojectionStride
			self._x_feature_shape = None
			self._y_feature_shape = None
			self._x_feature_size = None
		self._y_feature_size = self.embedding_dim

		if(self.using_input_image_projection):
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
					self._x_feature_shape = tuple(config.inputImageShape)
					self._x_feature_size = self._feature_size(self._x_feature_shape)
			else:
				self.input_projection = nn.Linear(config.inputLayerSize, self.embedding_dim, bias=False)
				self._initialise_random_linear(self.input_projection)
				self.input_projection.weight.requires_grad_(False)

		exemplar_flat_size = None
		if(self.use_target_exemplar_projection):
			exemplar_images = config.class_exemplar_images
			if exemplar_images is None:
				raise ValueError("targetProjectionExemplarImage=True requires exemplar images in the configuration.")
			if not isinstance(exemplar_images, pt.Tensor):
				exemplar_images = pt.as_tensor(exemplar_images, dtype=pt.float32)
			exemplar_images = exemplar_images.clone().detach().to(dtype=pt.float32)
			if exemplar_images.ndim != 4:
				raise ValueError("Class exemplar images must be a tensor of shape [num_classes, channels, height, width].")
			if exemplar_images.shape[0] != config.numberOfClasses:
				raise ValueError("Number of exemplar images must match number of classes.")
			self.register_buffer("class_exemplar_images", exemplar_images)
			exemplar_flat_size = self._feature_size(tuple(exemplar_images.shape[1:]))

		def _make_target_projection_module():
			if(self.using_target_image_projection):
				projection_module, _ = self._build_image_projection(config, stride=projection_stride, trainable=False)
				return projection_module
			if(self.use_target_exemplar_projection):
				assert exemplar_flat_size is not None
				linear_projection = self._build_linear_target_projection(exemplar_flat_size)
				return nn.Sequential(nn.Flatten(start_dim=1), linear_projection)
			return self._build_linear_target_projection(config.outputLayerSize)

		if self.target_projection_unique_per_layer:
			projection_count = self.recursion_steps
		else:
			projection_count = 1
		target_projection_modules = [_make_target_projection_module() for _ in range(projection_count)]
		self.target_projection_layers = nn.ModuleList(target_projection_modules)
		self.target_projection = self.target_projection_layers[-1]

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

		if(self.use_target_exemplar_projection):
			with pt.no_grad():
				weight_matrices = []
				for module in self.target_projection_layers:
					class_embeddings = self._project_exemplar_images(self.class_exemplar_images, module=module)
					weight_matrices.append(class_embeddings.transpose(0, 1).contiguous())
				weight_stack = pt.stack(weight_matrices, dim=0)
			if(self.train_classification_layer):
				self.target_projection_weight_stack = nn.Parameter(weight_stack)
				self.target_projection_weight_matrix = None
			else:
				self.register_buffer("target_projection_weight_stack", weight_stack)
				self.register_buffer("target_projection_weight_matrix", weight_stack[-1].clone())
		else:
			self.target_projection_weight_stack = None
			self.target_projection_weight_matrix = None

		self.recursive_layer = None
		self.recursive_cnn_layer = None
		self.recursive_ff_layer = None
		self.recursive_layer_schedule = None
		self.mlp_input_adapter = None

		cnn_layers = self.config.numberOfConvlayers
		ff_layers = self.config.numberOfFFLayers
		if(self.use_recursive_layers):
			if(self.using_rpi_cnn):
				layer_schedule = []
				self.recursive_cnn_layer = RecursiveCNNActionLayer(self._y_feature_shape, numberOfSublayers, layerScale, use_activation=self.use_hidden_activation, x_feature_shape=self._x_feature_shape)
				layer_schedule.extend(["cnn"] * cnn_layers)
				if(ff_layers > 0):
					self.recursive_ff_layer = RecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation)
					layer_schedule.extend(["ff"] * ff_layers)
				if(not layer_schedule):
					raise ValueError("useRPICNN=True requires numberOfConvlayers or numberOfFFLayers to be positive.")
				if(len(layer_schedule) != self.recursion_steps):
					raise ValueError(f"Configured numberOfLayers ({self.recursion_steps}) must equal numberOfConvlayers + numberOfFFLayers ({len(layer_schedule)}).")
				self.recursive_layer_schedule = layer_schedule
			else:
				self.recursive_layer = RecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation)
				self.recursive_layer_schedule = ["ff"] * self.recursion_steps
			self.nonrecursive_layers = None
		else:
			layer_modules = []
			if(self.using_rpi_cnn):
				layer_modules.extend([NonRecursiveCNNActionLayer(self._y_feature_shape, numberOfSublayers, layerScale, use_activation=self.use_hidden_activation, x_feature_shape=self._x_feature_shape) for _ in range(cnn_layers)])
				if(ff_layers > 0):
					layer_modules.extend([NonRecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation) for _ in range(ff_layers)])
			else:
				layer_modules.extend([NonRecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation) for _ in range(self.recursion_steps)])
			if(len(layer_modules) != self.recursion_steps):
				raise ValueError(f"Configured number of layers ({self.recursion_steps}) does not match instantiated layers ({len(layer_modules)}).")
			self.nonrecursive_layers = nn.ModuleList(layer_modules)

		if(useImageDataset):
			x_feature_size = self._x_feature_size
			target_feature_size = self._y_feature_size
			if(x_feature_size is not None and target_feature_size is not None):
				if(self.config.numberOfFFLayers > 0 and x_feature_size != target_feature_size):
					self.mlp_input_adapter = nn.Linear(x_feature_size, target_feature_size, bias=False)
					self._initialise_random_linear(self.mlp_input_adapter)
					self.mlp_input_adapter.weight.requires_grad_(False)
		if(useClassificationLayerLoss):
			self.embedding_loss_weight = 0.1

		self.lossFunctionFinal = nn.CrossEntropyLoss()
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		self.last_Z = None
		self.last_logits = None

	def _initialise_random_linear(self, module):
		std = 1.0 / math.sqrt(module.out_features)
		nn.init.normal_(module.weight, mean=0.0, std=std)

	def _initialise_random_conv(self, module):
		kernel_height, kernel_width = module.kernel_size
		fan_out = module.out_channels * kernel_height * kernel_width
		std = 1.0 / math.sqrt(fan_out)
		nn.init.normal_(module.weight, mean=0.0, std=std)

	def _build_linear_target_projection(self, in_features):
		linear_module = nn.Linear(in_features, self.embedding_dim, bias=False)
		self._initialise_random_linear(linear_module)
		linear_module.weight.requires_grad_(self.train_classification_layer)
		assert not self.using_target_image_projection
		if(targetProjectionSparse):
			self._apply_target_projection_sparsity(linear_module)
		return linear_module

	def _apply_target_projection_sparsity(self, module):
		if(targetProjectionSparsityLevel < 0.0 or targetProjectionSparsityLevel >= 1.0):
			raise ValueError("targetProjectionSparsityLevel must be in the range [0.0, 1.0).")
		if(targetProjectionSparsityLevel == 0.0):
			return
		with pt.no_grad():
			weight = module.weight
			num_elements = weight.numel()
			if(num_elements == 0):
				return
			zero_count = int(num_elements * targetProjectionSparsityLevel)
			if(zero_count <= 0):
				return
			flattened = weight.view(-1)
			zero_indices = pt.randperm(num_elements, device=weight.device)[:zero_count]
			flattened[zero_indices] = 0.0

	def _feature_size(self, feature_shape):
		channels, height, width = feature_shape
		return channels * height * width

	def _determine_cnn_feature_shape(self, config, stride):
		if(config.inputImageShape is None):
			raise ValueError("Image projection requires an input image shape.")
		number_of_convs = CNNprojectionNumlayers
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
		number_of_convs = CNNprojectionNumlayers

		input_channels, _, _ = config.inputImageShape
		feature_shape = self._determine_cnn_feature_shape(config, stride)
		out_channels, _, _ = feature_shape

		use_activation = CNNprojectionActivationFunction

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
		x_embed_base = self.encode_input(x)
		x_embed_raw, x_embed_mlp = self._prepare_dual_x_embed(x_embed_base)
		target_embeddings = self.encode_targets(y)
		final_step_index = max(0, self.recursion_steps - 1)

		if(trainOrTest and trainLocal):
			if(x_embed_mlp is x_embed_raw):
				x_embed_raw = x_embed_raw.detach()
				x_embed_mlp = x_embed_raw
			else:
				x_embed_raw = x_embed_raw.detach()
				x_embed_mlp = x_embed_mlp.detach() if x_embed_mlp is not None else None
			target_embeddings = target_embeddings.detach()
			Z = self._train_recursive_locally(x_embed_raw, x_embed_mlp, target_embeddings, y, optim, train_final_only=trainFinalIterationOnly)
			with pt.no_grad():
				final_target = self._target_embedding_for_layer(target_embeddings, final_step_index)
				total_loss, logits, classification_loss, embedding_alignment_loss, activated_Z = self._compute_total_loss(Z, final_target, y, layer_index=final_step_index)
			loss = total_loss.detach()
			accuracy = self.accuracyFunction(logits, y)
			self.last_Z = activated_Z.detach()
			self.last_logits = logits.detach()
			return loss, accuracy
		else:
			if(trainOrTest):
				x_embed_raw = x_embed_raw.requires_grad_()
				if(x_embed_mlp is not x_embed_raw and x_embed_mlp is not None):
					x_embed_mlp = x_embed_mlp.requires_grad_()
			target_embeddings = target_embeddings.detach()
			final_target = self._target_embedding_for_layer(target_embeddings, final_step_index)
			Z = self.iterate_prediction(x_embed_raw, x_embed_mlp, train_final_only=trainOrTest and trainFinalIterationOnly)
			total_loss, logits, _, _, activated_Z = self._compute_total_loss(Z, final_target, y, layer_index=final_step_index)
			accuracy = self.accuracyFunction(logits, y)
			self.last_Z = activated_Z.detach()
			self.last_logits = logits.detach()
			return total_loss, accuracy

	def _resolve_local_optimizer(self, optim, index):	
		return optim[index]
	
	def _prepare_dual_x_embed(self, x_embed):
		if(x_embed is None):
			return x_embed, x_embed
		if(x_embed.dim() != 2):
			return x_embed, x_embed
		if(x_embed.shape[1] == self.embedding_dim):
			return x_embed, x_embed
		if(self.mlp_input_adapter is None):
			return x_embed, x_embed
		projected = self.mlp_input_adapter(x_embed)
		projected = self.input_projection_activation(projected)
		projected = self.input_projection_activation_tanh(projected)
		return x_embed, projected

	def _select_x_embed(self, layer, x_embed_raw, x_embed_mlp):
		if(isinstance(layer, (RecursiveActionLayer, NonRecursiveActionLayer))):
			if(x_embed_mlp is not None):
				return x_embed_mlp
			return x_embed_raw
		return x_embed_raw

	def _resolve_recursive_layer_for_step(self, step):
		if(not self.use_recursive_layers):
			return None
		if(self.recursive_layer_schedule is None):
			return self.recursive_layer
		if(step < 0 or step >= len(self.recursive_layer_schedule)):
			raise IndexError(f"Recursive layer index {step} out of range for schedule length {len(self.recursive_layer_schedule)}.")
		layer_type = self.recursive_layer_schedule[step]
		if(layer_type == "cnn"):
			if(self.recursive_cnn_layer is None):
				raise ValueError("Recursive layer schedule references a CNN layer, but recursive_cnn_layer is not configured.")
			return self.recursive_cnn_layer
		if(layer_type == "ff"):
			if(self.recursive_ff_layer is not None):
				return self.recursive_ff_layer
			if(self.recursive_layer is not None):
				return self.recursive_layer
			raise ValueError("Recursive layer schedule references a feed-forward layer, but no recursive feed-forward layer is configured.")
		raise ValueError(f"Unsupported recursive layer type '{layer_type}'.")

	def _train_recursive_locally(self, x_embed_raw, x_embed_mlp, target_embeddings, y, optim, train_final_only=False):
		reference_embed = x_embed_mlp
		if(reference_embed is None):
			reference_embed = x_embed_raw
		if(initialiseZzero or reference_embed.shape[1] != self._y_feature_size):
			Z_state = self._zero_Z_like(reference_embed)
		else:
			Z_state = reference_embed
		update = None
		if(self.use_recursive_layers):
			final_step_index = max(0, self.recursion_steps - 1)
			for step in range(self.recursion_steps):
				layer = self._resolve_recursive_layer_for_step(step)
				x_for_layer = self._select_x_embed(layer, x_embed_raw, x_embed_mlp)
				if(train_final_only and step != final_step_index):
					with pt.no_grad():
						update = self._local_step_forward(Z_state, x_for_layer, layer_ref=layer, index=step)
				else:
					layer_target = self._target_embedding_for_layer(target_embeddings, step)
					update = self._local_step(Z_state, x_for_layer, layer_target, y, optim, index=step, layer_ref=layer)
				Z_state = self._applyResidual(Z_state, update)
		else:
			if(self.nonrecursive_layers is None or len(self.nonrecursive_layers) == 0):
				raise ValueError("Non-recursive training requested, but no non-recursive layers are configured.")
			final_step_index = len(self.nonrecursive_layers) - 1
			for step, layer in enumerate(self.nonrecursive_layers):
				x_for_layer = self._select_x_embed(layer, x_embed_raw, x_embed_mlp)
				if(train_final_only and step != final_step_index):
					with pt.no_grad():
						update = self._local_step_forward(Z_state, x_for_layer, layer_ref=layer, index=step)
				else:
					layer_target = self._target_embedding_for_layer(target_embeddings, step)
					update = self._local_step(Z_state, x_for_layer, layer_target, y, optim, index=step, layer_ref=layer)
				Z_state = self._applyResidual(Z_state, update)

		if(update is None):
			update = Z_state
		return update

	def _local_step(self, base, x_embed, target_embedding, y, optim, index, layer_ref=None):
		opt = self._resolve_local_optimizer(optim, index)
		Z = self._local_step_forward(base, x_embed, layer_ref, index=index)
		total_loss, *_ = self._compute_total_loss(Z, target_embedding, y, layer_index=index)
		opt.zero_grad()
		total_loss.backward()
		opt.step()
		Z = Z.detach()
		return Z

	def _local_step_forward(self, base, x_embed, layer_ref=None, index=None):
		if(self.use_recursive_layers):
			Z = self._iterate_forward_recursive(base, x_embed, layer_ref=layer_ref, index=index)
		else:
			Z = self._iterate_forward_nonrecursive(base, x_embed, layer_ref)
		return Z
	
	def _iterate_forward_recursive(self, base, x_embed, layer_ref=None, index=None):
		layer = layer_ref if layer_ref is not None else self.recursive_layer
		if(layer is None):
			raise ValueError("Recursive layer reference is not configured.")
		out = layer(base, x_embed)
		return out
	
	def _iterate_forward_nonrecursive(self, base, x_embed, layer_ref):
		out = layer_ref(base, x_embed)
		return out
	
	def iterate_prediction(self, x_embed_raw, x_embed_mlp=None, train_final_only=False):
		reference_embed = x_embed_mlp
		if(reference_embed is None):
			reference_embed = x_embed_raw
		if(initialiseZzero or reference_embed.shape[1] != self._y_feature_size):
			Z = self._zero_Z_like(reference_embed)
		else:
			Z = reference_embed
		if(self.use_recursive_layers):
			final_step_index = max(0, self.recursion_steps - 1)
			for step in range(self.recursion_steps):
				layer = self._resolve_recursive_layer_for_step(step)
				x_for_layer = self._select_x_embed(layer, x_embed_raw, x_embed_mlp)
				if(train_final_only and step != final_step_index):
					with pt.no_grad():
						update = layer(Z, x_for_layer)
				else:
					update = layer(Z, x_for_layer)
				Z = self._applyResidual(Z, update)
		else:
			if(self.nonrecursive_layers is None or len(self.nonrecursive_layers) == 0):
				raise ValueError("Non-recursive prediction requested, but no non-recursive layers are configured.")
			final_step_index = len(self.nonrecursive_layers) - 1
			for step, layer in enumerate(self.nonrecursive_layers):
				x_for_layer = self._select_x_embed(layer, x_embed_raw, x_embed_mlp)
				if(train_final_only and step != final_step_index):
					with pt.no_grad():
						update = layer(Z, x_for_layer)
				else:
					update = layer(Z, x_for_layer)
				Z = self._applyResidual(Z, update)
		return Z
	
	def _applyResidual(self, orig, update):
		if(layersFeedResidualInput):
			return orig + update
		else:
			return update
			
	def _compute_total_loss(self, Z, target_embedding, y, layer_index=None):
		logits = self.project_to_classes(Z, layer_index=layer_index)
		classification_loss = self.lossFunctionFinal(logits, y)
		embedding_alignment_loss = F.mse_loss(Z, target_embedding)
		if(useClassificationLayerLoss):
			if(useClassificationLayerLossStrict):
				total_loss = classification_loss
			else:
				total_loss = classification_loss + self.embedding_loss_weight * embedding_alignment_loss
		else:
			total_loss = embedding_alignment_loss
		return total_loss, logits, classification_loss, embedding_alignment_loss, Z

	def project_to_classes(self, Z, layer_index=None):
		# Using the frozen target projection weights as a random classifier head.
		layer_idx = self._resolve_projection_layer_index(layer_index)
		if(self.use_target_exemplar_projection):
			weight_stack = getattr(self, "target_projection_weight_stack", None)
			if(weight_stack is not None):
				max_index = weight_stack.shape[0] - 1
				layer_idx = max(0, min(layer_idx, max_index))
				weight = weight_stack[layer_idx]
			else:
				weight = self.target_projection_weight_matrix
		else:
			layer_idx = min(layer_idx, len(self.target_projection_layers) - 1)
			module = self.target_projection_layers[layer_idx]
			weight = self._extract_projection_weight(module)
		return pt.matmul(Z, weight)

	def _zero_Z_like(self, reference_tensor):
		return reference_tensor.new_zeros((reference_tensor.shape[0], self._y_feature_size))

	def _target_embedding_for_layer(self, target_embeddings, layer_index):
		if not isinstance(target_embeddings, pt.Tensor):
			return target_embeddings
		if(target_embeddings.dim() == 2):
			return target_embeddings
		if(target_embeddings.dim() != 3):
			raise ValueError("target_embeddings must be a tensor of shape [layers, batch, embedding_dim].")
		layer_count = target_embeddings.shape[0]
		if(layer_count == 0):
			raise ValueError("target_embeddings tensor must contain at least one layer.")
		if(layer_index is None):
			layer_index = layer_count - 1
		if(layer_index < 0):
			layer_index = layer_count + layer_index
		layer_index = max(0, min(layer_index, layer_count - 1))
		return target_embeddings[layer_index]

	def _resolve_projection_layer_index(self, layer_index):
		layer_count = len(self.target_projection_layers)
		if(layer_count == 0):
			raise ValueError("No target projection layers configured.")
		if(layer_index is None):
			return layer_count - 1
		if(layer_index < 0):
			layer_index = layer_count + layer_index
		return max(0, min(layer_index, layer_count - 1))

	def _extract_projection_weight(self, module):
		if isinstance(module, nn.Linear):
			return module.weight
		if isinstance(module, nn.Sequential):
			for submodule in reversed(list(module)):
				if isinstance(submodule, nn.Linear):
					return submodule.weight
		raise ValueError("Unable to locate linear weights in target projection module.")
	
	def encode_input(self, x):
		projected = self.encode_inputs(x)
		return projected

	def _project_exemplar_images(self, images, module=None):
		projection_module = module if module is not None else self.target_projection
		projected = projection_module(images)
		projected = self.target_projection_activation(projected)
		projected = self.target_projection_activation_tanh(projected)
		return projected
	
	def encode_targets(self, y):
		if(self.use_target_exemplar_projection):
			exemplar_images = self.class_exemplar_images.index_select(0, y)
			layer_embeddings = []
			for module in self.target_projection_layers:
				layer_embedding = self._project_exemplar_images(exemplar_images, module=module)
				layer_embeddings.append(layer_embedding.unsqueeze(0))
		else:
			y_one_hot = F.one_hot(y, num_classes=self.config.outputLayerSize).float()
			layer_embeddings = []
			for module in self.target_projection_layers:
				layer_embedding = module(y_one_hot)
				layer_embedding = self.target_projection_activation(layer_embedding)
				layer_embedding = self.target_projection_activation_tanh(layer_embedding)
				layer_embeddings.append(layer_embedding.unsqueeze(0))
		target_embeddings = pt.cat(layer_embeddings, dim=0)
		if not self.target_projection_unique_per_layer:
			target_embeddings = target_embeddings.expand(self.recursion_steps, -1, -1)
		return target_embeddings
	
	def encode_inputs(self, x):
		input_embedding = self.input_projection(x)
		input_embedding = self.input_projection_activation(input_embedding)
		input_embedding = self.input_projection_activation_tanh(input_embedding)
		return input_embedding
