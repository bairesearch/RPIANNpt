"""RPIANNpt_RPIANNmodelAutoencoder.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
RPIANNpt Recursive Prediction Improvement artificial neural network model autoencoder

"""

import torch as pt
from torch import nn
import torch.nn.functional as F
from ANNpt_globalDefs import *

def configure_input_projection_autoencoder(self, input_feature_size):
	linear_module = self._locate_linear_input_projection()
	if linear_module is None:
		raise ValueError("inputProjectionAutoencoder currently requires a linear input projection module.")
	self.input_projection_reverse = nn.Linear(self.embedding_dim, input_feature_size, bias=False)
	self._initialise_random_linear(self.input_projection_reverse)
	_set_module_requires_grad(self, self.input_projection, False)
	_set_module_requires_grad(self, self.input_projection_reverse, False)
	self.input_autoencoder_forward_optimizer = pt.optim.Adam(self.input_projection.parameters(), lr=learningRate)
	self.input_autoencoder_reverse_optimizer = None
	_update_input_projection_inverse(self)

def configure_target_projection_autoencoder(self, output_feature_size):
	reverse_modules = []
	for module in self.target_projection_layers:
		if not isinstance(module, nn.Linear):
			raise ValueError("targetProjectionAutoencoder currently requires linear target projection modules.")
		reverse_module = nn.Linear(self.embedding_dim, output_feature_size, bias=False)
		self._initialise_random_linear(reverse_module)
		reverse_modules.append(reverse_module)
	self.target_projection_reverse_layers = nn.ModuleList(reverse_modules)
	_set_modules_requires_grad(self, self.target_projection_reverse_layers, False)
	if(not self.train_classification_layer):
		_set_modules_requires_grad(self, self.target_projection_layers, False)
	forward_params = []
	for module in self.target_projection_layers:
		forward_params.extend(module.parameters())
	reverse_params = []
	for module in self.target_projection_reverse_layers:
		reverse_params.extend(module.parameters())
	self.target_autoencoder_forward_optimizer = pt.optim.Adam(forward_params, lr=learningRate)
	self.target_autoencoder_reverse_optimizer = pt.optim.Adam(reverse_params, lr=learningRate)
	_reset_target_autoencoder_grad_state(self)
	_update_target_projection_reverse_from_forward(self)
		  
def _set_module_requires_grad(self, module, requires_grad):
	if module is None:
		return
	for param in module.parameters():
		param.requires_grad_(requires_grad)

def _set_modules_requires_grad(self, modules, requires_grad):
	if modules is None:
		return
	if isinstance(modules, (list, tuple, nn.ModuleList)):
		for module in modules:
			_set_module_requires_grad(self, module, requires_grad)
	else:
		_set_module_requires_grad(self, modules, requires_grad)

def _apply_projection_autoencoder_noise(self, tensor):
	noise_std = getattr(self, "projection_autoencoder_noise_std", 0.0)
	if noise_std is None or noise_std <= 0.0:
		return tensor
	noise = pt.randn_like(tensor) * noise_std
	return tensor + noise

def train_autoencoders(self, x, y):
	if(self.useProjectionAutoencoderIndependent):
		return _train_autoencoders_independent(self, x, y)
	return _train_autoencoders_joint(self, x, y)

def _train_autoencoders_independent(self, x, y):
	phase = getattr(self, "projection_autoencoder_phase", "both")
	train_input = self.input_projection_autoencoder and (phase in ("input", "both"))
	train_target = self.target_projection_autoencoder and (phase in ("target", "both"))
	input_loss = None
	target_loss = None
	if(train_input):
		input_loss = _align_input_projection_with_targets(self, x, y)
	if(train_target):
		target_loss = _align_target_projection_with_inputs(self, x, y)
	return input_loss, target_loss

def _align_input_projection_with_targets(self, x, y):
	optimizer = getattr(self, "input_autoencoder_forward_optimizer", None)
	if(optimizer is None):
		return None
	_set_module_requires_grad(self, self.input_projection, True)
	_set_modules_requires_grad(self, self.target_projection_layers, False)
	optimizer.zero_grad(set_to_none=True)
	with pt.no_grad():
		target_embeddings = _reference_target_embedding(self, y)
	noisy_x = _apply_projection_autoencoder_noise(self, x)
	predicted = self.encode_inputs(noisy_x)
	loss = _projection_alignment_loss(self, predicted, target_embeddings, y)
	loss.backward()
	optimizer.step()
	_update_input_projection_inverse(self)
	_set_module_requires_grad(self, self.input_projection, False)
	_reset_target_autoencoder_grad_state(self)
	return loss.detach().item()

def _align_target_projection_with_inputs(self, x, y):
	optimizer = getattr(self, "target_autoencoder_forward_optimizer", None)
	if(optimizer is None):
		return None
	_set_module_requires_grad(self, self.input_projection, False)
	_set_modules_requires_grad(self, self.target_projection_layers, True)
	optimizer.zero_grad(set_to_none=True)
	with pt.no_grad():
		reference = _reference_input_embedding(self, x)
	y_vectors = _one_hot_targets(self, y)
	total_loss = 0.0
	layer_losses = 0
	for module in self.target_projection_layers:
		layer_embedding = module(y_vectors)
		layer_embedding = self.target_projection_activation(layer_embedding)
		layer_embedding = self.target_projection_activation_tanh(layer_embedding)
		layer_loss = _projection_alignment_loss(self, layer_embedding, reference, y)
		total_loss = total_loss + layer_loss
		layer_losses += 1
	if(layer_losses == 0):
		_reset_target_autoencoder_grad_state(self)
		return None
	loss = total_loss / float(layer_losses)
	loss.backward()
	optimizer.step()
	_reset_target_autoencoder_grad_state(self)
	_update_target_projection_reverse_from_forward(self)
	return loss.detach().item()

def _projection_alignment_loss(self, predicted, target, labels=None):
	loss = F.mse_loss(predicted, target)
	if(getattr(self, "use_projection_autoencoder_vicreg", False)):
		loss = loss + _projection_vicreg_penalty(self, predicted, target)
	if(getattr(self, "use_projection_autoencoder_vicreg_contrastive", False) and labels is not None):
		loss = loss + _projection_contrastive_loss(self, predicted, labels)
	return loss

def _projection_vicreg_penalty(self, tensor_a, tensor_b):
	if tensor_a is None or tensor_b is None:
		return 0.0
	lambda_weight = getattr(self, "projection_autoencoder_vicreg_lambda", 0.0)
	mu_weight = getattr(self, "projection_autoencoder_vicreg_mu", 0.0)
	nu_weight = getattr(self, "projection_autoencoder_vicreg_nu", 0.0)
	if(lambda_weight == 0.0 and mu_weight == 0.0 and nu_weight == 0.0):
		return 0.0
	invariance = F.mse_loss(tensor_a, tensor_b)
	variance = _vicreg_variance_term(self, tensor_a) + _vicreg_variance_term(self, tensor_b)
	covariance = _vicreg_covariance_term(self, tensor_a) + _vicreg_covariance_term(self, tensor_b)
	return lambda_weight * invariance + mu_weight * variance + nu_weight * covariance

def _vicreg_variance_term(self, tensor):
	if tensor.dim() < 2:
		return tensor.new_tensor(0.0)
	eps = getattr(self, "projection_autoencoder_vicreg_eps", 1e-4)
	variance = pt.var(tensor, dim=0, unbiased=False)
	std = pt.sqrt(variance + eps)
	return pt.mean(F.relu(1.0 - std))

def _vicreg_covariance_term(self, tensor):
	if tensor.dim() < 2:
		return tensor.new_tensor(0.0)
	batch_size = tensor.shape[0]
	if batch_size <= 1:
		return tensor.new_tensor(0.0)
	centered = tensor - tensor.mean(dim=0, keepdim=True)
	covariance = pt.matmul(centered.transpose(0, 1), centered) / float(batch_size - 1)
	diag = pt.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
	covariance = covariance * (1.0 - diag)
	return pt.sum(covariance.pow(2)) / covariance.shape[0]

def _projection_contrastive_loss(self, embeddings, labels):
	weight = getattr(self, "projection_autoencoder_contrastive_weight", 0.0)
	if weight is None or weight <= 0.0:
		return embeddings.new_tensor(0.0)
	if embeddings.dim() < 2:
		return embeddings.new_tensor(0.0)
	if labels is None:
		return embeddings.new_tensor(0.0)
	labels = labels.detach()
	batch_size = embeddings.shape[0]
	if batch_size <= 1:
		return embeddings.new_tensor(0.0)
	similarity = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
	device = embeddings.device
	diag = pt.eye(batch_size, device=device, dtype=pt.bool)
	label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
	same_pairs = label_matrix & (~diag)
	diff_pairs = (~label_matrix) & (~diag)
	pos_loss = similarity.new_tensor(0.0)
	neg_loss = similarity.new_tensor(0.0)
	if same_pairs.any():
		pos_loss = (1.0 - similarity[same_pairs]).mean()
	margin = getattr(self, "projection_autoencoder_contrastive_margin", 0.0)
	if diff_pairs.any():
		neg_loss = F.relu(similarity[diff_pairs] - margin).mean()
	return weight * (pos_loss + neg_loss)

def _train_autoencoders_joint(self, x, y):
	train_input = bool(self.input_projection_autoencoder)
	train_target = bool(self.target_projection_autoencoder)
	if((not train_input) and (not train_target)):
		return (None, None)
	input_optimizer = getattr(self, "input_autoencoder_forward_optimizer", None) if train_input else None
	target_optimizer = None
	use_reverse_decoders = self.target_projection_reverse_layers is not None and len(self.target_projection_reverse_layers) > 0
	if(train_target):
		if(use_reverse_decoders):
			target_optimizer = getattr(self, "target_autoencoder_reverse_optimizer", None)
			_set_modules_requires_grad(self, self.target_projection_reverse_layers, True)
		else:
			target_optimizer = getattr(self, "target_autoencoder_forward_optimizer", None)
			_set_modules_requires_grad(self, self.target_projection_layers, True)
	else:
		_set_modules_requires_grad(self, self.target_projection_reverse_layers, False)
		_set_modules_requires_grad(self, self.target_projection_layers, False)
	if(input_optimizer is None and target_optimizer is None):
		return (None, None)
	_set_module_requires_grad(self, self.input_projection, train_input)
	if(input_optimizer is not None):
		input_optimizer.zero_grad(set_to_none=True)
	if(target_optimizer is not None):
		target_optimizer.zero_grad(set_to_none=True)
	noisy_x = _apply_projection_autoencoder_noise(self, x)
	embeddings = self.encode_inputs(noisy_x)
	y_vectors = _one_hot_targets(self, y)
	layer_predictions = []
	if(use_reverse_decoders and self.target_projection_reverse_layers is not None and len(self.target_projection_reverse_layers) > 0):
		for module in self.target_projection_reverse_layers:
			layer_predictions.append(module(embeddings))
	else:
		for module in self.target_projection_layers:
			weight = self._extract_projection_weight(module)
			layer_predictions.append(pt.matmul(embeddings, weight))
	if(len(layer_predictions) == 0):
		_set_module_requires_grad(self, self.input_projection, False)
		_reset_target_autoencoder_grad_state(self)
		return (None, None)
	average_prediction = sum(layer_predictions) / float(len(layer_predictions))
	loss = F.mse_loss(average_prediction, y_vectors)
	if(self.use_projection_autoencoder_vicreg):
		target_embeddings = self.encode_targets(y)
		if isinstance(target_embeddings, pt.Tensor) and target_embeddings.dim() == 3:
			target_embeddings = self._target_embedding_for_layer(target_embeddings, -1)
		vicreg_penalty = _projection_vicreg_penalty(self, embeddings, target_embeddings)
		loss = loss + vicreg_penalty
	if(getattr(self, "use_projection_autoencoder_vicreg_contrastive", False)):
		loss = loss + _projection_contrastive_loss(self, embeddings, y)
	loss.backward()
	if(input_optimizer is not None):
		input_optimizer.step()
	if(target_optimizer is not None):
		target_optimizer.step()
	_reset_target_autoencoder_grad_state(self)
	_update_input_projection_inverse(self)
	if(use_reverse_decoders and self.target_projection_reverse_layers is not None):
		_update_target_projection_forward_from_reverse(self)
	else:
		_update_target_projection_reverse_from_forward(self)
	_set_module_requires_grad(self, self.input_projection, False)
	loss_value = loss.detach().item()
	input_loss = loss_value if input_optimizer is not None else None
	target_loss = loss_value if target_optimizer is not None else None
	return input_loss, target_loss

def _reference_target_embedding(self, y):
	target_embeddings = self.encode_targets(y)
	if isinstance(target_embeddings, pt.Tensor) and target_embeddings.dim() == 3:
		target_embeddings = self._target_embedding_for_layer(target_embeddings, -1)
	return target_embeddings.detach()

def _reference_input_embedding(self, x):
	noisy_x = _apply_projection_autoencoder_noise(self, x)
	input_embedding = self.encode_inputs(noisy_x)
	return input_embedding.detach()

def _one_hot_targets(self, y):
	return F.one_hot(y, num_classes=self.config.outputLayerSize).float().to(y.device)

def _update_input_projection_inverse(self):
	if(self.input_projection_reverse is None):
		return
	linear_module = self._locate_linear_input_projection()
	if(linear_module is None):
		return
	with pt.no_grad():
		weight = linear_module.weight
		pinv = pt.linalg.pinv(weight)
		pinv = pinv.to(self.input_projection_reverse.weight.dtype)
		self.input_projection_reverse.weight.copy_(pinv)

def _update_target_projection_reverse_from_forward(self):
	if(self.target_projection_reverse_layers is None):
		return
	with pt.no_grad():
		for module, reverse in zip(self.target_projection_layers, self.target_projection_reverse_layers):
			weight = self._extract_projection_weight(module)
			pinv = pt.linalg.pinv(weight)
			pinv = pinv.to(reverse.weight.dtype)
			reverse.weight.copy_(pinv)
	_set_modules_requires_grad(self, self.target_projection_reverse_layers, False)

def _update_target_projection_forward_from_reverse(self):
	if(self.target_projection_reverse_layers is None):
		return
	with pt.no_grad():
		for module, reverse in zip(self.target_projection_layers, self.target_projection_reverse_layers):
			reverse_weight = reverse.weight
			pinv = pt.linalg.pinv(reverse_weight)
			target_weight = self._extract_projection_weight(module)
			target_weight.copy_(pinv.to(target_weight.dtype))

def _reset_target_autoencoder_grad_state(self):
	if(self.target_projection_layers is not None):
		_set_modules_requires_grad(self, self.target_projection_layers, self.train_classification_layer)
	if(self.target_projection_autoencoder):
		_set_modules_requires_grad(self, self.target_projection_reverse_layers, False)
