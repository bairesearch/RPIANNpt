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
AEANNpt Autoencoder/Breakaway generated artificial neural network model

"""

import math

import torch as pt
from torch import nn
import torch.nn.functional as F
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy

class AEANNconfig():
	def __init__(self, batchSize, numberOfLayers, numberOfConvlayers, hiddenLayerSize, CNNhiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples):
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

class RecursiveActionLayer(nn.Module):
	def __init__(self, embedding_dim, hidden_dim=None, action_scale=0.25):
		super().__init__()
		if(hidden_dim is None):
			hidden_dim = embedding_dim * 2
		self.action = nn.Sequential(
			nn.Linear(embedding_dim * 2, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, embedding_dim)
		)
		self.action_scale = action_scale

	def forward(self, x_embed, y_hat):
		combined = pt.cat([x_embed, y_hat], dim=-1)
		# Limit the magnitude of the proposed update so the recursion remains stable
		update = pt.tanh(self.action(combined)) * self.action_scale
		return update

class AEANNmodel(nn.Module):
		
	def __init__(self, config):
		super().__init__()
		self.config = config

		# Interpret the existing hidden layer size as the shared embedding dimensionality.
		self.embedding_dim = config.hiddenLayerSize
		self.recursion_steps = max(1, config.numberOfLayers)

		self.input_projection = nn.Linear(config.inputLayerSize, self.embedding_dim, bias=False)
		self.target_projection = nn.Linear(config.outputLayerSize, self.embedding_dim, bias=False)
		self._initialise_random_projection(self.input_projection)
		self._initialise_random_projection(self.target_projection)
		self.input_projection.weight.requires_grad_(False)
		self.target_projection.weight.requires_grad_(False)

		self.initial_predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
		self.recursive_layer = RecursiveActionLayer(self.embedding_dim)
		self.embedding_loss_weight = 0.1

		self.lossFunctionFinal = nn.CrossEntropyLoss()
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		self.last_y_hat = None
		self.last_logits = None

	def _initialise_random_projection(self, module):
		std = 1.0 / math.sqrt(module.out_features)
		nn.init.normal_(module.weight, mean=0.0, std=std)

	def _compute_total_loss(self, y_hat, target_embedding, y):
		logits = self.project_to_classes(y_hat)
		classification_loss = self.lossFunctionFinal(logits, y)
		embedding_alignment_loss = F.mse_loss(y_hat, target_embedding)
		total_loss = classification_loss + self.embedding_loss_weight * embedding_alignment_loss
		return total_loss, logits, classification_loss, embedding_alignment_loss

	def encode_input(self, x):
		if(useImageDataset):
			batch_size = x.shape[0]
			x = x.reshape(batch_size, -1)
		return self.input_projection(x)

	def encode_targets(self, y):
		y_one_hot = F.one_hot(y, num_classes=self.config.outputLayerSize).float()
		return self.target_projection(y_one_hot)

	def initialise_prediction(self, x_embed):
		return pt.tanh(self.initial_predictor(x_embed))

	def iterate_prediction(self, x_embed, y_hat):
		for _ in range(self.recursion_steps):
			y_hat = y_hat + self.recursive_layer(x_embed, y_hat)
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

		for step in range(self.recursion_steps):
			base_state = y_hat_state
			def iterate_forward(base=base_state):
				return base + self.recursive_layer(x_embed, base)
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
