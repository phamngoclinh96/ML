'''
Miscellaneous utility functions
'''
import collections
import configparser
import glob
import operator
import os
import random
import time
import datetime
import shutil
from pprint import pprint
import distutils.util as distutils_util

import brat2conll
import conll2brat
import utils_nlp


def order_dictionary(dictionary, mode, reverse=False):
    '''
    Order a dictionary by 'key' or 'value'.
    mode should be either 'key' or 'value'
    http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    '''

    if mode =='key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(0),
                                              reverse=reverse))
    elif mode =='value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(1),
                                              reverse=reverse))
    elif mode =='key_value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              reverse=reverse))
    elif mode =='value_key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=lambda x: (x[1], x[0]),
                                              reverse=reverse))
    else:
        raise ValueError("Unknown mode. Should be 'key' or 'value'")

def reverse_dictionary(dictionary):
    '''
    http://stackoverflow.com/questions/483666/python-reverse-inverse-a-mapping
    http://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain
    '''
    #print('type(dictionary): {0}'.format(type(dictionary)))
    if type(dictionary) is collections.OrderedDict:
        #print(type(dictionary))
        return collections.OrderedDict([(v, k) for k, v in dictionary.items()])
    else:
        return {v: k for k, v in dictionary.items()}

def merge_dictionaries(*dict_args):
    '''
    http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def pad_list(old_list, padding_size, padding_value):
    '''
    http://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
    Example: pad_list([6,2,3], 5, 0) returns [6,2,3,0,0]
    '''
    assert padding_size >= len(old_list)
    return old_list + [padding_value] * (padding_size-len(old_list))

def get_basename_without_extension(filepath):
    '''
    Getting the basename of the filepath without the extension
    E.g. 'data/formatted/movie_reviews.pickle' -> 'movie_reviews'
    '''
    return os.path.basename(os.path.splitext(filepath)[0])

def create_folder_if_not_exists(directory):
    '''
    Create the folder if it doesn't exist already.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_current_milliseconds():
    '''
    http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    '''
    return(int(round(time.time() * 1000)))


def get_current_time_in_seconds():
    '''
    http://stackoverflow.com/questions/415511/how-to-get-current-time-in-python
    '''
    return(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

def get_current_time_in_miliseconds():
    '''
    http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    '''
    return(get_current_time_in_seconds() + '-' + str(datetime.datetime.now().microsecond))


def convert_configparser_to_dictionary(config):
    '''
    http://stackoverflow.com/questions/1773793/convert-configparser-items-to-dictionary
    '''
    my_config_parser_dict = {s:dict(config.items(s)) for s in config.sections()}
    return my_config_parser_dict

def get_parameter_to_section_of_configparser(config):
    parameter_to_section = {}
    for s in config.sections():
        for p, _ in config.items(s):
            parameter_to_section[p] = s
    return parameter_to_section


def copytree(src, dst, symlinks=False, ignore=None):
    '''
    http://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
    '''
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


#####################################################################################


def create_stats_graph_folder(parameters):
        # Initialize stats_graph_folder
        experiment_timestamp = get_current_time_in_miliseconds()
        dataset_name = get_basename_without_extension(parameters['dataset_text_folder'])
        model_name = '{0}_{1}'.format(dataset_name, experiment_timestamp)
        create_folder_if_not_exists(parameters['output_folder'])
        stats_graph_folder = os.path.join(parameters['output_folder'], model_name)  # Folder where to save graphs
        create_folder_if_not_exists(stats_graph_folder)
        return stats_graph_folder, experiment_timestamp


def load_parameters(parameters_filepath, arguments={}, verbose=True):
    '''
    Load parameters from the ini file if specified, take into account any command line argument, and ensure that each parameter is cast to the correct type.
    Command line arguments take precedence over parameters specified in the parameter file.
    '''
    parameters = {'pretrained_model_folder': '../model',
                  'dataset_text_folder': '../data/en',
                  'character_embedding_dimension': 25,
                  'character_lstm_hidden_state_dimension': 25,
                  'check_for_digits_replaced_with_zeros': True,
                  'check_for_lowercase': True,
                  'debug': False,
                  'dropout_rate': 0.5,
                  'experiment_name': 'test',
                  'freeze_token_embeddings': False,
                  'gradient_clipping_value': 5.0,
                  'learning_rate': 0.005,
                  'load_only_pretrained_token_embeddings': False,
                  'load_all_pretrained_token_embeddings': False,
                  'main_evaluation_mode': 'conll',
                  'maximum_number_of_epochs': 100,
                  'number_of_cpu_threads': 8,
                  'number_of_gpus': 0,
                  'optimizer': 'sgd',
                  'output_folder': '../output',
                  'patience': 10,
                  'plot_format': 'pdf',
                  'reload_character_embeddings': True,
                  'reload_character_lstm': True,
                  'reload_crf': True,
                  'reload_feedforward': True,
                  'reload_token_embeddings': True,
                  'reload_token_lstm': True,
                  'remap_unknown_tokens_to_unk': True,
                  'spacylanguage': 'en',
                  'tagging_format': 'bioes',
                  'token_embedding_dimension': 100,
                  'token_lstm_hidden_state_dimension': 100,
                  'token_pretrained_embedding_filepath': '../embedding/glove.6B.100d.txt',
                  'tokenizer': 'spacy',
                  'train_model': True,
                  'use_character_lstm': True,
                  'use_crf': True,
                  'use_pretrained_model': False,
                  'verbose': False}
    # If a parameter file is specified, load it
    if len(parameters_filepath) > 0:
        conf_parameters = configparser.ConfigParser()
        conf_parameters.read(parameters_filepath)
        nested_parameters = convert_configparser_to_dictionary(conf_parameters)
        for k, v in nested_parameters.items():
            parameters.update(v)
    # Ensure that any arguments the specified in the command line overwrite parameters specified in the parameter file
    for k, v in arguments.items():
        if arguments[k] != arguments['argument_default_value']:
            parameters[k] = v
    for k, v in parameters.items():
        v = str(v)
        # If the value is a list delimited with a comma, choose one element at random.
        if ',' in v:
            v = random.choice(v.split(','))
            parameters[k] = v
        # Ensure that each parameter is cast to the correct type
        if k in ['character_embedding_dimension', 'character_lstm_hidden_state_dimension', 'token_embedding_dimension',
                 'token_lstm_hidden_state_dimension', 'patience', 'maximum_number_of_epochs', 'maximum_training_time',
                 'number_of_cpu_threads', 'number_of_gpus']:
            parameters[k] = int(v)
        elif k in ['dropout_rate', 'learning_rate', 'gradient_clipping_value']:
            parameters[k] = float(v)
        elif k in ['remap_unknown_tokens_to_unk', 'use_character_lstm', 'use_crf', 'train_model',
                   'use_pretrained_model', 'debug', 'verbose',
                   'reload_character_embeddings', 'reload_character_lstm', 'reload_token_embeddings',
                   'reload_token_lstm', 'reload_feedforward', 'reload_crf',
                   'check_for_lowercase', 'check_for_digits_replaced_with_zeros', 'freeze_token_embeddings',
                   'load_only_pretrained_token_embeddings', 'load_all_pretrained_token_embeddings']:
            parameters[k] = distutils_util.strtobool(v)
    # If loading pretrained model, set the model hyperparameters according to the pretraining parameters
    if parameters['use_pretrained_model']:
        pretraining_parameters = \
        load_parameters(parameters_filepath=os.path.join(parameters['pretrained_model_folder'], 'parameters.ini'),
                         verbose=False)[0]
        for name in ['use_character_lstm', 'character_embedding_dimension', 'character_lstm_hidden_state_dimension',
                     'token_embedding_dimension', 'token_lstm_hidden_state_dimension', 'use_crf']:
            if parameters[name] != pretraining_parameters[name]:
                print(
                    'WARNING: parameter {0} was overwritten from {1} to {2} to be consistent with the pretrained model'.format(
                        name, parameters[name], pretraining_parameters[name]))
                parameters[name] = pretraining_parameters[name]
    if verbose: pprint(parameters)
    # Update conf_parameters to reflect final parameter values
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(os.path.join('test', 'test-parameters-training.ini'))
    parameter_to_section = get_parameter_to_section_of_configparser(conf_parameters)
    for k, v in parameters.items():
        conf_parameters.set(parameter_to_section[k], k, str(v))

    return parameters, conf_parameters


def get_valid_dataset_filepaths(parameters, dataset_types=['train', 'valid', 'test', 'deploy']):
    dataset_filepaths = {}
    dataset_brat_folders = {}
    for dataset_type in dataset_types:
        dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_text_folder'],
                                                       '{0}.txt'.format(dataset_type))
        dataset_brat_folders[dataset_type] = os.path.join(parameters['dataset_text_folder'], dataset_type)
        dataset_compatible_with_brat_filepath = os.path.join(parameters['dataset_text_folder'],
                                                             '{0}_compatible_with_brat.txt'.format(dataset_type))

        # Conll file exists
        if os.path.isfile(dataset_filepaths[dataset_type]) and os.path.getsize(dataset_filepaths[dataset_type]) > 0:
            # Brat text files exist
            if os.path.exists(dataset_brat_folders[dataset_type]) and len(
                    glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:

                # Check compatibility between conll and brat files
                brat2conll.check_brat_annotation_and_text_compatibility(dataset_brat_folders[dataset_type])
                if os.path.exists(dataset_compatible_with_brat_filepath):
                    dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath
                conll2brat.check_compatibility_between_conll_and_brat_text(dataset_filepaths[dataset_type],
                                                                           dataset_brat_folders[dataset_type])

            # Brat text files do not exist
            else:

                # Populate brat text and annotation files based on conll file
                conll2brat.conll_to_brat(dataset_filepaths[dataset_type], dataset_compatible_with_brat_filepath,
                                         dataset_brat_folders[dataset_type], dataset_brat_folders[dataset_type])
                dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

        # Conll file does not exist
        else:
            # Brat text files exist
            if os.path.exists(dataset_brat_folders[dataset_type]) and len(
                    glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:
                dataset_filepath_for_tokenizer = os.path.join(parameters['dataset_text_folder'],
                                                              '{0}_{1}.txt'.format(dataset_type,
                                                                                   parameters['tokenizer']))
                if os.path.exists(dataset_filepath_for_tokenizer):
                    conll2brat.check_compatibility_between_conll_and_brat_text(dataset_filepath_for_tokenizer,
                                                                               dataset_brat_folders[
                                                                                   dataset_type])
                else:
                    # Populate conll file based on brat files
                    brat2conll.brat_to_conll(dataset_brat_folders[dataset_type], dataset_filepath_for_tokenizer,
                                             parameters['tokenizer'], parameters['spacylanguage'])
                dataset_filepaths[dataset_type] = dataset_filepath_for_tokenizer

            # Brat text files do not exist
            else:
                del dataset_filepaths[dataset_type]
                del dataset_brat_folders[dataset_type]
                continue

        if parameters['tagging_format'] == 'bioes':
            # Generate conll file with BIOES format
            bioes_filepath = os.path.join(parameters['dataset_text_folder'], '{0}_bioes.txt'.format(
                get_basename_without_extension(dataset_filepaths[dataset_type])))
            utils_nlp.convert_conll_from_bio_to_bioes(dataset_filepaths[dataset_type], bioes_filepath)
            dataset_filepaths[dataset_type] = bioes_filepath

    return dataset_filepaths, dataset_brat_folders


def check_parameter_compatiblity(parameters, dataset_filepaths):
    # Check mode of operation
    if parameters['train_model']:
        if 'train' not in dataset_filepaths or 'valid' not in dataset_filepaths:
            raise IOError(
                "If train_model is set to True, both train and valid set must exist in the specified dataset folder: {0}".format(
                    parameters['dataset_text_folder']))
    elif parameters['use_pretrained_model']:
        if 'train' in dataset_filepaths and 'valid' in dataset_filepaths:
            print(
                "WARNING: train and valid set exist in the specified dataset folder, but train_model is set to FALSE: {0}".format(
                    parameters['dataset_text_folder']))
        if 'test' not in dataset_filepaths and 'deploy' not in dataset_filepaths:
            raise IOError(
                "For prediction mode, either test set and deploy set must exist in the specified dataset folder: {0}".format(
                    parameters['dataset_text_folder']))
    else:  # if not parameters['train_model'] and not parameters['use_pretrained_model']:
        raise ValueError('At least one of train_model and use_pretrained_model must be set to True.')

    if parameters['use_pretrained_model']:
        if all([not parameters[s] for s in
                ['reload_character_embeddings', 'reload_character_lstm', 'reload_token_embeddings',
                 'reload_token_lstm', 'reload_feedforward', 'reload_crf']]):
            raise ValueError(
                'If use_pretrained_model is set to True, at least one of reload_character_embeddings, reload_character_lstm, reload_token_embeddings, reload_token_lstm, reload_feedforward, reload_crf must be set to True.')

    if parameters['gradient_clipping_value'] < 0:
        parameters['gradient_clipping_value'] = abs(parameters['gradient_clipping_value'])
