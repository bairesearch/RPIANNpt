"""RPIANNpt_RPIANNmodelLayerMLP.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
RPIANNpt Recursive Prediction Improvement artificial neural network model layer MLP

"""

import math

import torch as pt
from torch import nn
import torch.nn.functional as F
from ANNpt_globalDefs import *

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

