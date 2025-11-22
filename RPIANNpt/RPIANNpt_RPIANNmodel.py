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
#if(useCNNprojection):
import RPIANNpt_RPIANNmodelCNNprojection
if(useRPICNN):
	from RPIANNpt_RPIANNmodelLayerCNN import NonRecursiveCNNActionLayer, RecursiveCNNActionLayer
if(useProjectionAutoencoder):
	import RPIANNpt_RPIANNmodelAutoencoder


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
		if(self.using_input_image_projection):
			self.use_linear_input_projection = False
		else:
			self.use_linear_input_projection = bool(useInputProjection)
		self.useProjectionAutoencoder = bool(useProjectionAutoencoder)
		if(self.useProjectionAutoencoder):
			self.useProjectionAutoencoderIndependent = bool(useProjectionAutoencoderIndependent)
			self.input_projection_autoencoder = bool(inputProjectionAutoencoder)
			self.input_autoencoder_independent = bool(inputProjectionAutoencoderIndependent)
			self.target_projection_autoencoder = bool(targetProjectionAutoencoder)
			self.target_autoencoder_independent = bool(targetProjectionAutoencoderIndependent)
			self.projection_autoencoder_independent_separate = bool(projectionAutoencoderIndependentSeparateTrainPhases)
			self.projection_autoencoder_phase = "both"
			self.projection_autoencoder_warmup_epochs = projectionAutoencoderWarmupEpochs
			self.projection_autoencoder_noise_std = projectionAutoencoderDenoisingStd
			self.projection_autoencoder_pretrain_epochs = projectionAutoencoderPretrainEpochs
			self.use_projection_autoencoder_vicreg = bool(projectionAutoencoderVICReg)
			self.use_projection_autoencoder_vicreg_contrastive = bool(projectionAutoencoderVICRegContrastiveLoss)
			if(projectionAutoencoderVICReg):
				self.projection_autoencoder_vicreg_lambda = projectionAutoencoderVICRegLambda
				self.projection_autoencoder_vicreg_mu = projectionAutoencoderVICRegMu
				self.projection_autoencoder_vicreg_nu = projectionAutoencoderVICRegNu
				self.projection_autoencoder_vicreg_eps = projectionAutoencoderVICRegEps
			if(projectionAutoencoderVICRegContrastiveLoss):
				self.projection_autoencoder_contrastive_weight = projectionAutoencoderVICRegContrastiveWeight
				self.projection_autoencoder_contrastive_margin = projectionAutoencoderVICRegContrastiveMargin
			self.current_epoch = 0
			self.input_projection_reverse = None
			self.input_autoencoder_forward_optimizer = None
			self.input_autoencoder_reverse_optimizer = None
			self.target_projection_reverse_layers = None
			self.target_autoencoder_forward_optimizer = None
			self.target_autoencoder_reverse_optimizer = None
			if(self.input_projection_autoencoder and (not self.use_linear_input_projection)):
				raise ValueError("inputProjectionAutoencoder=True requires useInputProjection=True and useCNNinputProjection=False.")
			if(self.target_projection_autoencoder and (self.use_target_exemplar_projection or self.using_target_image_projection)):
				raise ValueError("targetProjectionAutoencoder=True currently requires linear target projections (useCNNtargetProjection=False and targetProjectionExemplarImage=False).")
		if(useImageDataset):	
			projection_stride = CNNprojectionStride
			self._x_feature_shape = None
			self._y_feature_shape = None
			self._x_feature_size = None
		self._y_feature_size = self.embedding_dim

		if(self.using_input_image_projection):
			if(self.using_rpi_cnn):
				self.input_projection, feature_shape = RPIANNpt_RPIANNmodelCNNprojection.build_image_projection(self, config, stride=projection_stride, trainable=False)
				self._x_feature_shape = feature_shape
				self._y_feature_shape = RPIANNpt_RPIANNmodelCNNprojection.determine_cnn_feature_shape(self, config, projection_stride)
				self._x_feature_size = self._feature_size(self._x_feature_shape)
				self._y_feature_size = self._feature_size(self._y_feature_shape)
			else:
				self.input_projection, feature_shape = RPIANNpt_RPIANNmodelCNNprojection.build_image_projection(self, config, stride=projection_stride, trainable=False)
				self._x_feature_shape = feature_shape
				self._y_feature_shape = feature_shape
				self._x_feature_size = self._feature_size(feature_shape)
				self._y_feature_size = self._x_feature_size
		else:
			if(useImageDataset):	
				self._x_feature_shape = tuple(config.inputImageShape)
				input_flat_features = self._feature_size(self._x_feature_shape)
				if(self.using_rpi_cnn):
					self._y_feature_shape = RPIANNpt_RPIANNmodelCNNprojection.determine_cnn_feature_shape(self, config, projection_stride)
					self._x_feature_size = self._feature_size(self._x_feature_shape)
					self._y_feature_size = self._feature_size(self._y_feature_shape)
				else:
					self._x_feature_size = input_flat_features
				if(self.use_linear_input_projection):
					if(self.using_rpi_cnn):
						raise ValueError("useInputProjection=True with useRPICNN=True and useCNNinputProjection=False is not supported.")
					flatten_module = nn.Flatten(start_dim=1)
					linear_module = nn.Linear(input_flat_features, self.embedding_dim, bias=False)
					self._initialise_random_linear(linear_module)
					self.input_projection = nn.Sequential(flatten_module, linear_module)
					if(self.useProjectionAutoencoder and self.input_projection_autoencoder):
						RPIANNpt_RPIANNmodelAutoencoder.configure_input_projection_autoencoder(self, input_flat_features)
					else:
						linear_module.weight.requires_grad_(False)
					self._x_feature_size = self.embedding_dim
				else:
					self.input_projection = nn.Sequential(nn.Flatten(start_dim=1))
			else:
				if(self.use_linear_input_projection):
					self.input_projection = nn.Linear(config.inputLayerSize, self.embedding_dim, bias=False)
					self._initialise_random_linear(self.input_projection)
					self._x_feature_size = self.embedding_dim
					if(self.useProjectionAutoencoder and self.input_projection_autoencoder):
						RPIANNpt_RPIANNmodelAutoencoder.configure_input_projection_autoencoder(self, config.inputLayerSize)
					else:
						self.input_projection.weight.requires_grad_(False)
				else:
					self.input_projection = nn.Identity()
					self._x_feature_size = config.inputLayerSize

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
				projection_module, _ = RPIANNpt_RPIANNmodelCNNprojection.build_image_projection(self, config, stride=projection_stride, trainable=False)
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
		if(self.useProjectionAutoencoder and self.target_projection_autoencoder):
			RPIANNpt_RPIANNmodelAutoencoder.configure_target_projection_autoencoder(self, config.outputLayerSize)

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
					class_embeddings = RPIANNpt_RPIANNmodelCNNprojection.project_exemplar_images(self, self.class_exemplar_images, module=module)
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

		cnn_layers = self.config.numberOfConvlayers
		ff_layers = self.config.numberOfFFLayers
		if(self.use_recursive_layers):
			if(self.using_rpi_cnn):
				layer_schedule = []
				self.recursive_cnn_layer = RecursiveCNNActionLayer(self._y_feature_shape, numberOfSublayers, layerScale, use_activation=self.use_hidden_activation, x_feature_shape=self._x_feature_shape)
				layer_schedule.extend(["cnn"] * cnn_layers)
				if(ff_layers > 0):
					self.recursive_ff_layer = RecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation, x_feature_size=self._x_feature_size)
					layer_schedule.extend(["ff"] * ff_layers)
				if(not layer_schedule):
					raise ValueError("useRPICNN=True requires numberOfConvlayers or numberOfFFLayers to be positive.")
				if(len(layer_schedule) != self.recursion_steps):
					raise ValueError(f"Configured numberOfLayers ({self.recursion_steps}) must equal numberOfConvlayers + numberOfFFLayers ({len(layer_schedule)}).")
				self.recursive_layer_schedule = layer_schedule
			else:
				self.recursive_layer = RecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation, x_feature_size=self._x_feature_size)
				self.recursive_layer_schedule = ["ff"] * self.recursion_steps
			self.nonrecursive_layers = None
		else:
			layer_modules = []
			if(self.using_rpi_cnn):
				layer_modules.extend([NonRecursiveCNNActionLayer(self._y_feature_shape, numberOfSublayers, layerScale, use_activation=self.use_hidden_activation, x_feature_shape=self._x_feature_shape) for _ in range(cnn_layers)])
				if(ff_layers > 0):
					layer_modules.extend([NonRecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation, x_feature_size=self._x_feature_size) for _ in range(ff_layers)])
			else:
				layer_modules.extend([NonRecursiveActionLayer(self.embedding_dim, layerScale, use_activation=self.use_hidden_activation, x_feature_size=self._x_feature_size) for _ in range(self.recursion_steps)])
			if(len(layer_modules) != self.recursion_steps):
				raise ValueError(f"Configured number of layers ({self.recursion_steps}) does not match instantiated layers ({len(layer_modules)}).")
			self.nonrecursive_layers = nn.ModuleList(layer_modules)

		if(useClassificationLayerLoss):
			self.embedding_loss_weight = 0.1

		self.lossFunctionFinal = nn.CrossEntropyLoss()
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		self.last_Z = None
		self.last_logits = None

	def set_training_epoch(self, epoch):
		self.current_epoch = epoch
	
	def set_projection_autoencoder_phase(self, phase):
		self.projection_autoencoder_phase = phase
	
	def reset_projection_autoencoder_phase(self):
		self.projection_autoencoder_phase = "both"

	def pretrain_projection_autoencoders(self, x, y):
		if(not self.useProjectionAutoencoder):
			return (None, None)
		return RPIANNpt_RPIANNmodelAutoencoder.train_autoencoders(self, x, y)

	def _should_run_projection_autoencoders(self):
		if(not self.useProjectionAutoencoder):
			return False
		pretrain_epochs = getattr(self, "projection_autoencoder_pretrain_epochs", 0)
		if(pretrain_epochs is not None and pretrain_epochs > 0):
			# When a dedicated pretraining stage is configured, skip autoencoder updates during main training.
			return False
		warmup_limit = getattr(self, "projection_autoencoder_warmup_epochs", 0)
		if(warmup_limit is None or warmup_limit <= 0):
			return True
		current_epoch = getattr(self, "current_epoch", 0)
		return current_epoch < warmup_limit

	def _initialise_random_linear(self, module):
		std = 1.0 / math.sqrt(module.out_features)
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
			
	def forward(self, trainOrTest, x, y, optim, layer=None):
		if(trainOrTest and self._should_run_projection_autoencoders()):
			RPIANNpt_RPIANNmodelAutoencoder.train_autoencoders(self, x, y)
		x_embed = self.encode_input(x)
		target_embeddings = self.encode_targets(y)
		final_step_index = max(0, self.recursion_steps - 1)

		if(trainOrTest and trainLocal):
			x_embed = x_embed.detach()
			target_embeddings = target_embeddings.detach()
			Z = self._iterate_local(x_embed, target_embeddings, y, optim, train_final_only=trainFinalIterationOnly)
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
				x_embed = x_embed.requires_grad_()
			target_embeddings = target_embeddings.detach()
			final_target = self._target_embedding_for_layer(target_embeddings, final_step_index)
			Z = self._iterate_nonlocal(x_embed, train_final_only=trainOrTest and trainFinalIterationOnly)
			total_loss, logits, _, _, activated_Z = self._compute_total_loss(Z, final_target, y, layer_index=final_step_index)
			accuracy = self.accuracyFunction(logits, y)
			self.last_Z = activated_Z.detach()
			self.last_logits = logits.detach()
			return total_loss, accuracy

	def _resolve_local_optimizer(self, optim, index):	
		return optim[index]
	
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

	def _iterate_local(self, x_embed, target_embeddings, y, optim, train_final_only=False):
		reference_embed = x_embed
		if(initialiseZzero or reference_embed.shape[1] != self._y_feature_size):
			Z_state = self._zero_Z_like(reference_embed)
		else:
			Z_state = reference_embed
		update = None
		if(self.use_recursive_layers):
			final_step_index = max(0, self.recursion_steps - 1)
			for step in range(self.recursion_steps):
				layer = self._resolve_recursive_layer_for_step(step)
				x_for_layer = x_embed
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
				x_for_layer = x_embed
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
	
	def _iterate_nonlocal(self, x_embed, train_final_only=False):
		reference_embed = x_embed
		if(initialiseZzero or reference_embed.shape[1] != self._y_feature_size):
			Z = self._zero_Z_like(reference_embed)
		else:
			Z = reference_embed
		if(self.use_recursive_layers):
			final_step_index = max(0, self.recursion_steps - 1)
			for step in range(self.recursion_steps):
				layer = self._resolve_recursive_layer_for_step(step)
				x_for_layer = x_embed
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
				x_for_layer = x_embed
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

	def _locate_linear_input_projection(self):
		module = self.input_projection
		if isinstance(module, nn.Linear):
			return module
		if isinstance(module, nn.Sequential):
			for submodule in reversed(list(module)):
				if isinstance(submodule, nn.Linear):
					return submodule
		return None
	
	def encode_input(self, x):
		projected = self.encode_inputs(x)
		return projected
	
	def encode_targets(self, y):
		if(self.use_target_exemplar_projection):
			exemplar_images = self.class_exemplar_images.index_select(0, y)
			layer_embeddings = []
			for module in self.target_projection_layers:
				layer_embedding = RPIANNpt_RPIANNmodelCNNprojection.project_exemplar_images(self, exemplar_images, module=module)
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
