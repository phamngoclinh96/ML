from __future__ import print_function
import os
import argparse
from argparse import RawTextHelpFormatter
import sys
from ModelEntity import NeuroNER

import warnings

pretrained_model_folder='../model'
dataset_text_folder='../data/en'
character_embedding_dimension=25,
character_lstm_hidden_state_dimension=25,
check_for_digits_replaced_with_zeros=True,
check_for_lowercase=True,
debug=False,
dropout_rate=0.5,
experiment_name='experiment',
freeze_token_embeddings=False,
gradient_clipping_value=5.0,
learning_rate=0.005,
load_only_pretrained_token_embeddings=False,
main_evaluation_mode='conll',
maximum_number_of_epochs=10,
number_of_cpu_threads=4,
number_of_gpus=0,
optimizer='sgd',
output_folder='../output',
patience=10,
plot_format='pdf',
reload_character_embeddings=True,
reload_character_lstm=True,
reload_crf=True,
reload_feedforward=True,
reload_token_embeddings=True,
reload_token_lstm=True,
remap_unknown_tokens_to_unk=True,
spacylanguage='en',
tagging_format='bioes',
token_embedding_dimension=100,
token_lstm_hidden_state_dimension=100,
token_pretrained_embedding_filepath='../embedding/glove.6B.100d.txt',
tokenizer='spacy',
train_model=True,
use_character_lstm=True,
use_crf=True,
use_pretrained_model=False,
verbose=False


nn = NeuroNER(pretrained_model_folder='../model',dataset_text_folder=dataset_text_folder,maximum_number_of_epochs=10,use_pretrained_model=True,debug=True)

nn.predict('From 1987 until 2000 NIPS was held in Denver, United States. Since then, the conference was held in Vancouver, Canada (2001–2010), Granada, Spain (2011), and Lake Tahoe, United States (2012–2013)')
