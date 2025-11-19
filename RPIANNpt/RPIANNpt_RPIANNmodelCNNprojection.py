"""RPIANNpt_RPIANNmodelCNNprojection.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
RPIANNpt Recursive Prediction Improvement artificial neural network model CNN projection

"""

import torch as pt
from torch import nn
import torch.nn.functional as F
from ANNpt_globalDefs import *
import math

def determine_cnn_feature_shape(self, config, stride):
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

def build_image_projection(self, config, stride=2, trainable=False):
	if(config.inputImageShape is None):
		raise ValueError("Image projection requires a defined input image shape.")
	if(stride not in (1, 2)):
		raise ValueError("Image projection stride must be 1 or 2.")
	number_of_convs = CNNprojectionNumlayers

	input_channels, _, _ = config.inputImageShape
	feature_shape = determine_cnn_feature_shape(self, config, stride)
	out_channels, _, _ = feature_shape

	use_activation = CNNprojectionActivationFunction

	layers = []
	in_channels = input_channels
	for _ in range(number_of_convs):
		conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		_initialise_random_conv(self, conv)
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

def _initialise_random_conv(self, module):
	kernel_height, kernel_width = module.kernel_size
	fan_out = module.out_channels * kernel_height * kernel_width
	std = 1.0 / math.sqrt(fan_out)
	nn.init.normal_(module.weight, mean=0.0, std=std)

def project_exemplar_images(self, images, module=None):
	projection_module = module if module is not None else self.target_projection
	projected = projection_module(images)
	projected = self.target_projection_activation(projected)
	projected = self.target_projection_activation_tanh(projected)
	return projected

