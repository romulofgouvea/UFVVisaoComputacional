#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:03:57 2017

@author: romulo
"""

import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, save_checkpoints=False, tensorboard_dir='/home/user/tensorlogs', report_rate=20)

#==============================================================================
# # 3 channels for colour, 1 channel for greyscale
# channels = 3
# 
# # Setup and hyperparameters
# model.set_batch_size(128)
# model.set_image_dimensions(256, 256, channels)
# model.set_learning_rate(0.001)
# model.set_maximum_training_epochs(700)
# model.set_train_test_split(0.75)
#==============================================================================

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(4)
model.set_number_of_threads(8)
model.set_image_dimensions(128, 128, channels)
model.set_resize_images(True)


model.set_problem_type('regression')
model.set_num_regression_outputs(1)
model.set_train_test_split(0.8)
model.set_learning_rate(0.0001)
model.set_weight_initializer('xavier')
model.set_maximum_training_epochs(500)

# Augmentation options
model.set_augmentation_brightness_and_contrast(True)
model.set_augmentation_flip_horizontal(True)
model.set_augmentation_flip_vertical(True)
model.set_augmentation_crop(True)

# Load all data for IPPN leaf counting dataset
model.load_ippn_leaf_count_dataset_from_directory('./home/romulo/Documents/Arquivos/UFV/6 - Periodo/SIN 393 - INTRODUÇÃO À VISÃO COMPUTACIONAL/Projeto Artigo/Conjunto de dados/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon')

# Specify pre-processing steps
model.add_preprocessing_step('auto-segmentation')

# Define a model architecture
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[5, 5, channels, 32], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_output_layer()

# Train!
model.begin_training()