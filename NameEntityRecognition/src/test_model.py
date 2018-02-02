import matplotlib

matplotlib.use('Agg')
import train
import dataset as ds
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from lstm_crf import EntityLSTM
import utils
import os
import conll2brat
import glob
import codecs
import shutil
import time
import copy
import evaluate
import random
import pickle
import brat2conll
import numpy as np
import utils_nlp
import distutils.util as distutils_util
import configparser
from pprint import pprint

parameters = {'pretrained_model_folder':'../model',
                      'dataset_text_folder':'../../../ML_EntityData/data/en',
                      'character_embedding_dimension':25,
                      'character_lstm_hidden_state_dimension':25,
                      'check_for_digits_replaced_with_zeros':True,
                      'check_for_lowercase':True,
                      'debug':False,
                      'dropout_rate':0.5,
                      'experiment_name':'test',
                      'freeze_token_embeddings':False,
                      'gradient_clipping_value':5.0,
                      'learning_rate':0.005,
                      'load_only_pretrained_token_embeddings':False,
                      'load_all_pretrained_token_embeddings':False,
                      'main_evaluation_mode':'conll',
                      'maximum_number_of_epochs':3,
                      'number_of_cpu_threads':8,
                      'number_of_gpus':0,
                      'optimizer':'sgd',
                      'output_folder':'../../../ML_EntityData/output',
                      'patience':10,
                      'plot_format':'pdf',
                      'reload_character_embeddings':True,
                      'reload_character_lstm':True,
                      'reload_crf':True,
                      'reload_feedforward':True,
                      'reload_token_embeddings':True,
                      'reload_token_lstm':True,
                      'remap_unknown_tokens_to_unk':True,
                      'spacylanguage':'en',
                      'tagging_format':'bioes',
                      'token_embedding_dimension':100,
                      'token_lstm_hidden_state_dimension':100,
                      'token_pretrained_embedding_filepath':'../../../ML_EntityData/embedding/glove.6B.100d.txt',
                      'tokenizer':'spacy',
                      'train_model':True,
                      'use_character_lstm':True,
                      'use_crf':True,
                      'use_pretrained_model':False,
                      'verbose':False}


# Load dataset
dataset_filepaths, dataset_brat_folders = utils.get_valid_dataset_filepaths(parameters)
dataset = ds.Dataset(verbose=False, debug=False)
token_to_vector = dataset.load_dataset(dataset_filepaths, parameters)



# Create model lstm+crf
session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
            inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
            device_count={'CPU': 1, 'GPU': parameters['number_of_gpus']},
            allow_soft_placement=True,
            # automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
            log_device_placement=False
        )
sess = tf.Session(config=session_conf)

with sess.as_default():
    # Create model and initialize or load pretrained model
    ### Instantiate the model
    model = EntityLSTM(dataset=dataset, token_embedding_dimension=parameters['token_embedding_dimension'],
                       character_lstm_hidden_state_dimension=parameters['character_lstm_hidden_state_dimension'],
                       token_lstm_hidden_state_dimension=parameters['token_lstm_hidden_state_dimension'],
                       character_embedding_dimension=parameters['character_embedding_dimension'],
                       use_crf=parameters['use_crf'],
                       use_character_lstm=parameters['use_character_lstm'],
                       gradient_clipping_value=parameters['gradient_clipping_value'],
                       learning_rate=parameters['learning_rate'],
                       freeze_token_embeddings=parameters['freeze_token_embeddings'],
                       optimizer=parameters['optimizer'],
                       maximum_number_of_epochs=parameters['maximum_number_of_epochs'])

sess.run(tf.global_variables_initializer())

# Load embedding
model.load_pretrained_token_embeddings(sess, dataset,embedding_filepath=parameters['token_pretrained_embedding_filepath'],
                                                       check_lowercase= parameters['check_for_lowercase'],check_digits=parameters['check_for_digits_replaced_with_zeros'],
                                                       token_to_vector=token_to_vector)
# Initial params_train
transition_params_trained = np.random.rand(len(dataset.unique_labels) + 2,len(dataset.unique_labels) + 2)
# Restore model trained
# transition_params_trained = model.restore_from_pretrained_model(dataset, sess , model_pathfile=os.path.join(parameters['pretrained_model_folder'],'model.ckpt'),
#                                                                                      dataset_pathfile=(parameters['pretrained_model_folder']+'/dataset.pickle'),
#                                                                                      embedding_filepath= parameters['token_pretrained_embedding_filepath'],
#                                                                                      character_dimension = parameters['character_embedding_dimension'],
#                                                                                      token_dimension=parameters['token_embedding_dimension'],token_to_vector=token_to_vector)
del token_to_vector

# Train model



stats_graph_folder, experiment_timestamp = utils.create_stats_graph_folder(parameters)

        # Initialize and save execution details
start_time = time.time()
results = {}
results['epoch'] = {}
results['execution_details'] = {}
results['execution_details']['train_start'] = start_time
results['execution_details']['time_stamp'] = experiment_timestamp
results['execution_details']['early_stop'] = False
results['execution_details']['keyboard_interrupt'] = False
results['execution_details']['num_epochs'] = 0
results['model_options'] = copy.copy(parameters)

model_folder = os.path.join(stats_graph_folder, 'model')
utils.create_folder_if_not_exists(model_folder)

pickle.dump(dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))

tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
utils.create_folder_if_not_exists(tensorboard_log_folder)
tensorboard_log_folders = {}
for dataset_type in dataset_filepaths.keys():
    tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 'tensorboard_logs', dataset_type)
    utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])

# Instantiate the writers for TensorBoard
writers = {}
for dataset_type in dataset_filepaths.keys():
    writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type], graph=sess.graph)
embedding_writer = tf.summary.FileWriter(
    model_folder)  # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings

embeddings_projector_config = projector.ProjectorConfig()
tensorboard_token_embeddings = embeddings_projector_config.embeddings.add()
tensorboard_token_embeddings.tensor_name = model.token_embedding_weights.name
token_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_tokens.tsv')
tensorboard_token_embeddings.metadata_path = os.path.relpath(token_list_file_path, '..')

tensorboard_character_embeddings = embeddings_projector_config.embeddings.add()
tensorboard_character_embeddings.tensor_name = model.character_embedding_weights.name
character_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_characters.tsv')
tensorboard_character_embeddings.metadata_path = os.path.relpath(character_list_file_path, '..')

projector.visualize_embeddings(embedding_writer, embeddings_projector_config)

# Write metadata for TensorBoard embeddings
token_list_file = codecs.open(token_list_file_path, 'w', 'UTF-8')
for token_index in range(dataset.vocabulary_size):
    token_list_file.write('{0}\n'.format(dataset.index_to_token[token_index]))
token_list_file.close()

character_list_file = codecs.open(character_list_file_path, 'w', 'UTF-8')
for character_index in range(dataset.alphabet_size):
    if character_index == dataset.PADDING_CHARACTER_INDEX:
        character_list_file.write('PADDING\n')
    else:
        character_list_file.write('{0}\n'.format(dataset.index_to_character[character_index]))
character_list_file.close()

# Start training + evaluation loop. Each iteration corresponds to 1 epoch.
bad_counter = 0  # number of epochs with no improvement on the validation test in terms of F1-score
previous_best_valid_f1_score = 0
epoch_number = -1
try:
    while True:

        step = 0
        epoch_number += 1
        print('\nStarting epoch {0}'.format(epoch_number))

        epoch_start_time = time.time()

        if epoch_number != 0:
            # Train model: loop over all sequences of training set with shuffling
            sequence_numbers = list(range(len(dataset.token_indices['train'])))
            random.shuffle(sequence_numbers)
            for sequence_number in sequence_numbers:
                transition_params_trained = train.train_step(sess, dataset, sequence_number, model, parameters['dropout_rate'])
                step += 1
                if step % 10 == 0:
                    print('Training {0:.2f}% done'.format(step / len(sequence_numbers) * 100), end='\r', flush=True)

        epoch_elapsed_training_time = time.time() - epoch_start_time
        print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time), flush=True)

        y_pred, y_true, output_filepaths = train.predict_labels_lite(sess=sess,model= model,transition_params_trained= transition_params_trained,
                                                                         dataset=dataset,epoch_number= epoch_number,
                                                                        stats_graph_folder= stats_graph_folder,dataset_filepaths= dataset_filepaths,
                                                                        tagging_format= parameters['tagging_format'], main_evaluation_mode=parameters['main_evaluation_mode'],use_crf=parameters['use_crf'])

        # # Evaluate model: save and plot results
        # evaluate.evaluate_model(results, dataset, y_pred, y_true, stats_graph_folder, epoch_number,
        #                                 epoch_start_time, output_filepaths, parameters)
        #
        # if parameters['use_pretrained_model'] and not parameters['train_model']:
        #     conll2brat.output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder)
        #     break
        #
        # # Save model
        model.saver.save(sess, os.path.join(model_folder, 'model_{0:05d}.ckpt'.format(epoch_number)))
        #
        # # Save TensorBoard logs
        # summary = sess.run(model.summary_op, feed_dict=None)
        # writers['train'].add_summary(summary, epoch_number)
        # writers['train'].flush()
        # utils.copytree(writers['train'].get_logdir(), model_folder)
        #
        # # Early stop
        # valid_f1_score = results['epoch'][epoch_number][0]['valid']['f1_score']['micro']
        # if valid_f1_score > previous_best_valid_f1_score:
        #     bad_counter = 0
        #     previous_best_valid_f1_score = valid_f1_score
        #     conll2brat.output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder,
        #                                       overwrite=True)
        #     transition_params_trained = transition_params_trained
        # else:
        #     bad_counter += 1
        # print("The last {0} epochs have not shown improvements on the validation set.".format(bad_counter))
        #
        # if bad_counter >= parameters['patience']:
        #     print('Early Stop!')
        #     results['execution_details']['early_stop'] = True
        #     break

        if epoch_number >= parameters['maximum_number_of_epochs']: break


except KeyboardInterrupt:
    results['execution_details']['keyboard_interrupt'] = True
    print('Training interrupted')

print('Finishing the experiment')
end_time = time.time()
results['execution_details']['train_duration'] = end_time - start_time
results['execution_details']['train_end'] = end_time
evaluate.save_results(results, stats_graph_folder)
for dataset_type in dataset_filepaths.keys():
    writers[dataset_type].close()




# End train




prediction_count=0


def predict(text):
    #         if prediction_count == 1:
    parameters['dataset_text_folder'] = os.path.join('..', 'data', 'temp')
    stats_graph_folder, _ = utils.create_stats_graph_folder(parameters)

    # Update the deploy folder, file, and dataset
    dataset_type = 'deploy'
    ### Delete all deployment data
    for filepath in glob.glob(os.path.join(parameters['dataset_text_folder'], '{0}*'.format(dataset_type))):
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)
        else:
            os.remove(filepath)
    ### Create brat folder and file
    dataset_brat_deploy_folder = os.path.join(parameters['dataset_text_folder'], dataset_type)
    utils.create_folder_if_not_exists(dataset_brat_deploy_folder)
    dataset_brat_deploy_filepath = os.path.join(dataset_brat_deploy_folder, 'temp_{0}.txt'.format(
        str(prediction_count).zfill(5)))  # self._get_dataset_brat_deploy_filepath(dataset_brat_deploy_folder)
    with codecs.open(dataset_brat_deploy_filepath, 'w', 'UTF-8') as f:
        f.write(text)
    ### Update deploy filepaths
    dataset_filepaths, dataset_brat_folders = utils.get_valid_dataset_filepaths(parameters,
                                                                           dataset_types=[dataset_type])
    dataset_filepaths.update(dataset_filepaths)
    dataset_brat_folders.update(dataset_brat_folders)
    ### Update the dataset for the new deploy set
    dataset.update_dataset(dataset_filepaths, [dataset_type])

    # Predict labels and output brat
    output_filepaths = {}
    prediction_output = train.prediction_step(sess, dataset, dataset_type, model,
                                              transition_params_trained, stats_graph_folder,
                                              prediction_count, dataset_filepaths, parameters['tagging_format'],
                                              parameters['main_evaluation_mode'])
    _, _, output_filepaths[dataset_type] = prediction_output
    conll2brat.output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder, overwrite=True)

    # Print and output result
    text_filepath = os.path.join(stats_graph_folder, 'brat', 'deploy',
                                 os.path.basename(dataset_brat_deploy_filepath))
    annotation_filepath = os.path.join(stats_graph_folder, 'brat', 'deploy', '{0}.ann'.format(
        utils.get_basename_without_extension(dataset_brat_deploy_filepath)))
    text2, entities = brat2conll.get_entities_from_brat(text_filepath, annotation_filepath, verbose=True)
    assert (text == text2)
    return entities


print(predict('my name is Ngoc Linh'))
