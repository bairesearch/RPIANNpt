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
	self.input_autoencoder_reverse_optimizer = pt.optim.Adam(self.input_projection_reverse.parameters(), lr=learningRate)

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
	input_loss = None
	target_loss = None
	if(self.use_input_projection_autoencoder):
		input_loss = _train_input_autoencoder(self, x)
	if(self.use_target_projection_autoencoder):
		target_loss = _train_target_autoencoder(self, y)
	return input_loss, target_loss

def _train_input_autoencoder(self, x):
	if(self.input_projection_reverse is None):
		return
	if(self.input_autoencoder_forward_optimizer is None or self.input_autoencoder_reverse_optimizer is None):
		return
	x_batch = x.detach()
	losses = []
	if(self.input_autoencoder_independent):
		losses.append(_execute_input_autoencoder_round(self, x_batch, train_forward=True))
		losses.append(_execute_input_autoencoder_round(self, x_batch, train_forward=False))
	else:
		losses.append(_execute_input_autoencoder_joint(self, x_batch))
	_set_module_requires_grad(self, self.input_projection, False)
	_set_module_requires_grad(self, self.input_projection_reverse, False)
	losses = [l for l in losses if l is not None]
	if(len(losses) == 0):
		return None
	return sum(losses) / float(len(losses))

def _execute_input_autoencoder_round(self, x, train_forward):
	if(train_forward):
		_set_module_requires_grad(self, self.input_projection, True)
		_set_module_requires_grad(self, self.input_projection_reverse, False)
		optimizer = self.input_autoencoder_forward_optimizer
		detach_encoder = False
	else:
		_set_module_requires_grad(self, self.input_projection, False)
		_set_module_requires_grad(self, self.input_projection_reverse, True)
		optimizer = self.input_autoencoder_reverse_optimizer
		detach_encoder = True
	if(optimizer is None):
		return None
	optimizer.zero_grad(set_to_none=True)
	loss = _input_autoencoder_loss(self, x, detach_encoder=detach_encoder)
	if(loss is None):
		return None
	loss.backward()
	optimizer.step()
	return loss.detach().item()

def _execute_input_autoencoder_joint(self, x):
	if(self.input_autoencoder_forward_optimizer is None or self.input_autoencoder_reverse_optimizer is None):
		return None
	_set_module_requires_grad(self, self.input_projection, True)
	_set_module_requires_grad(self, self.input_projection_reverse, True)
	self.input_autoencoder_forward_optimizer.zero_grad(set_to_none=True)
	self.input_autoencoder_reverse_optimizer.zero_grad(set_to_none=True)
	loss = _input_autoencoder_loss(self, x, detach_encoder=False)
	if(loss is None):
		return None
	loss.backward()
	self.input_autoencoder_forward_optimizer.step()
	self.input_autoencoder_reverse_optimizer.step()
	return loss.detach().item()

def _input_autoencoder_loss(self, x, detach_encoder):
	if(detach_encoder):
		with pt.no_grad():
			noisy_x = _apply_projection_autoencoder_noise(self, x)
			latent = self.encode_inputs(noisy_x)
		latent = latent.detach()
	else:
		noisy_x = _apply_projection_autoencoder_noise(self, x)
		latent = self.encode_inputs(noisy_x)
	reconstructed = self.input_projection_reverse(latent)
	target = _flatten_autoencoder_target(x)
	return F.mse_loss(reconstructed, target)

def _flatten_autoencoder_target(x):
	if(x.dim() <= 2):
		return x
	batch_size = x.shape[0]
	return x.reshape(batch_size, -1)

def _train_target_autoencoder(self, y):
	if(self.target_projection_reverse_layers is None):
		return None
	if(self.target_autoencoder_forward_optimizer is None or self.target_autoencoder_reverse_optimizer is None):
		return None
	one_hot = F.one_hot(y, num_classes=self.config.outputLayerSize).float().to(y.device)
	one_hot = one_hot.detach()
	losses = []
	if(self.target_autoencoder_independent):
		losses.append(_execute_target_autoencoder_round(self, one_hot, train_forward=True))
		losses.append(_execute_target_autoencoder_round(self, one_hot, train_forward=False))
	else:
		losses.append(_execute_target_autoencoder_joint(self, one_hot))
	_reset_target_autoencoder_grad_state(self)
	losses = [l for l in losses if l is not None]
	if(len(losses) == 0):
		return None
	return sum(losses) / float(len(losses))

def _execute_target_autoencoder_round(self, y_vectors, train_forward):
	if(train_forward):
		_set_modules_requires_grad(self, self.target_projection_layers, True)
		_set_modules_requires_grad(self, self.target_projection_reverse_layers, False)
		optimizer = self.target_autoencoder_forward_optimizer
		detach_encoder = False
	else:
		_set_modules_requires_grad(self, self.target_projection_layers, False)
		_set_modules_requires_grad(self, self.target_projection_reverse_layers, True)
		optimizer = self.target_autoencoder_reverse_optimizer
		detach_encoder = True
	if(optimizer is None):
		return None
	optimizer.zero_grad(set_to_none=True)
	loss = _target_autoencoder_loss(self, y_vectors, detach_encoder=detach_encoder)
	if(loss is None):
		return None
	loss.backward()
	optimizer.step()
	return loss.detach().item()

def _execute_target_autoencoder_joint(self, y_vectors):
	if(self.target_autoencoder_forward_optimizer is None or self.target_autoencoder_reverse_optimizer is None):
		return None
	_set_modules_requires_grad(self, self.target_projection_layers, True)
	_set_modules_requires_grad(self, self.target_projection_reverse_layers, True)
	self.target_autoencoder_forward_optimizer.zero_grad(set_to_none=True)
	self.target_autoencoder_reverse_optimizer.zero_grad(set_to_none=True)
	loss = _target_autoencoder_loss(self, y_vectors, detach_encoder=False)
	if(loss is None):
		return None
	loss.backward()
	self.target_autoencoder_forward_optimizer.step()
	self.target_autoencoder_reverse_optimizer.step()
	return loss.detach().item()

def _target_autoencoder_loss(self, y_vectors, detach_encoder):
	if(self.target_projection_reverse_layers is None):
		return None
	total_loss = 0.0
	layer_count = len(self.target_projection_reverse_layers)
	if(layer_count == 0):
		return None
	for module, reverse_module in zip(self.target_projection_layers, self.target_projection_reverse_layers):
		if(detach_encoder):
			with pt.no_grad():
				noisy_targets = _apply_projection_autoencoder_noise(self, y_vectors)
				embedding = _encode_target_with_module(self, module, noisy_targets)
			embedding = embedding.detach()
		else:
			noisy_targets = _apply_projection_autoencoder_noise(self, y_vectors)
			embedding = _encode_target_with_module(self, module, noisy_targets)
		reconstructed = reverse_module(embedding)
		total_loss = total_loss + F.mse_loss(reconstructed, y_vectors)
	return total_loss / float(layer_count)

def _encode_target_with_module(self, module, y_vectors):
	layer_embedding = module(y_vectors)
	layer_embedding = self.target_projection_activation(layer_embedding)
	layer_embedding = self.target_projection_activation_tanh(layer_embedding)
	return layer_embedding

def _reset_target_autoencoder_grad_state(self):
	if(not self.use_target_projection_autoencoder):
		return
	_set_modules_requires_grad(self, self.target_projection_layers, self.train_classification_layer)
	_set_modules_requires_grad(self, self.target_projection_reverse_layers, False)
