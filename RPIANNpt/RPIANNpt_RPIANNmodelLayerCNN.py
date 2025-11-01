"""RPIANNpt_RPIANNmodelLayerCNN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
RPIANNpt Recursive Prediction Improvement artificial neural network model layer CNN

"""

import math

import torch as pt
from torch import nn
import torch.nn.functional as F
from ANNpt_globalDefs import *
from torch.nn.modules.utils import _pair

class LocallyConnected2d(nn.Module):
	def __init__(self, in_channels, out_channels, output_size, kernel_size, stride=1, padding=0, bias=True):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = _pair(kernel_size)
		self.stride = _pair(stride)
		self.padding = _pair(padding)
		self.output_height, self.output_width = output_size
		kernel_height, kernel_width = self.kernel_size
		location_count = self.output_height * self.output_width
		self.weight = nn.Parameter(pt.empty(location_count, self.out_channels, self.in_channels * kernel_height * kernel_width))
		if(bias):
			self.bias = nn.Parameter(pt.empty(location_count, self.out_channels))
		else:
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self):
		kernel_height, kernel_width = self.kernel_size
		fan_out = self.out_channels * kernel_height * kernel_width
		std = 1.0 / math.sqrt(fan_out)
		nn.init.normal_(self.weight, mean=0.0, std=std)
		if(self.bias is not None):
			nn.init.zeros_(self.bias)

	def forward(self, input_tensor):
		batch_size, _, input_height, input_width = input_tensor.shape
		stride_height, stride_width = self.stride
		pad_height, pad_width = self.padding
		kernel_height, kernel_width = self.kernel_size
		expected_height = (input_height + 2 * pad_height - kernel_height) // stride_height + 1
		expected_width = (input_width + 2 * pad_width - kernel_width) // stride_width + 1
		if(expected_height != self.output_height or expected_width != self.output_width):
			raise ValueError("LocallyConnected2d received input with unexpected spatial dimensions.")
		patches = F.unfold(input_tensor, kernel_size=self.kernel_size, dilation=1, padding=self.padding, stride=self.stride)
		patches = patches.transpose(1, 2)	#shape: [batch, locations, in_channels*kernel_height*kernel_width]
		locations = self.output_height * self.output_width
		if(patches.shape[1] != locations):
			raise ValueError("Unfold produced mismatched number of spatial locations.")
		output = pt.einsum('blf,lcf->bcl', patches, self.weight)
		if(self.bias is not None):
			output = output + self.bias.transpose(0, 1).unsqueeze(0)
		output = output.view(batch_size, self.out_channels, self.output_height, self.output_width)
		return output

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
		self.use_unique_pixel_weights = RPICNNuniqueWeightsPerPixel

		if(layersFeedConcatInput):
			conv_in_channels = self.x_channels + self.y_channels
		else:
			conv_in_channels = self.y_channels
		conv_out_channels = self.y_channels
		self.conv_kernel_size = 3
		self.conv_stride = 1
		self.conv_padding = 1

		self.conv_layers = nn.ModuleList()
		self.pool_layers = nn.ModuleList()
		for _ in range(self.number_of_conv_layers):
			if(self.use_unique_pixel_weights):
				conv = LocallyConnected2d(conv_in_channels, conv_out_channels, (self.feature_height, self.feature_width), kernel_size=self.conv_kernel_size, stride=self.conv_stride, padding=self.conv_padding, bias=True)
			else:
				conv = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=self.conv_kernel_size, stride=self.conv_stride, padding=self.conv_padding, bias=True)
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
