import os
import tensorflow as tf
import numpy as np
import sklearn.metrics
from evaluate import remap_labels
import pickle
import utils_tf
import codecs
import utils_nlp
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def train_step(sess, dataset, sequence_number, model, dropout_rate=0.5):
    # Perform one iteration
    token_indices_sequence = dataset.token_indices['train'][sequence_number]
    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and np.random.uniform() < 0.5:
            token_indices_sequence[i] = dataset.UNK_TOKEN_INDEX
    feed_dict = {
      model.input_token_indices: token_indices_sequence,
      model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
      model.input_token_character_indices: dataset.character_indices_padded['train'][sequence_number],
      model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
      model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
      model.dropout_keep_prob: 1-dropout_rate
    }
    _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_parameters],
                    feed_dict)
    return transition_params_trained



def prediction_step_lite(sess, dataset, dataset_type, model, transition_params_trained, stats_graph_folder,
                    epoch_number, dataset_filepaths,
                    tagging_format,main_evaluation_mode,
                    use_crf=True):
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type,epoch_number))
    output_file = codecs.open(output_filepath, 'w', 'UTF-8')
    original_conll_file = codecs.open(dataset_filepaths[dataset_type], 'r', 'UTF-8')

    for i in range(len(dataset.token_indices[dataset_type])):
        feed_dict = {
          model.input_token_indices: dataset.token_indices[dataset_type][i],
          model.input_token_character_indices: dataset.character_indices_padded[dataset_type][i],
          model.input_token_lengths: dataset.token_lengths[dataset_type][i],
          model.input_label_indices_vector: dataset.label_vector_indices[dataset_type][i],
          model.dropout_keep_prob: 1.
        }
        unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
        if use_crf:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
            predictions = predictions[1:-1]
        else:
            predictions = predictions.tolist()

        assert(len(predictions) == len(dataset.tokens[dataset_type][i]))
        output_string = ''
        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        gold_labels = dataset.labels[dataset_type][i]
        if tagging_format == 'bioes':
            prediction_labels = utils_nlp.bioes_to_bio(prediction_labels)
            gold_labels = utils_nlp.bioes_to_bio(gold_labels)
        for prediction, token, gold_label in zip(prediction_labels, dataset.tokens[dataset_type][i], gold_labels):
            while True:
                line = original_conll_file.readline()
                split_line = line.strip().split(' ')
                if '-DOCSTART-' in split_line[0] or len(split_line) == 0 or len(split_line[0]) == 0:
                    continue
                else:
                    token_original = split_line[0]
                    if tagging_format == 'bioes':
                        split_line.pop()
                    gold_label_original = split_line[-1]
                    assert(token == token_original and gold_label == gold_label_original)
                    break
            split_line.append(prediction)
            output_string += ' '.join(split_line) + '\n'
        output_file.write(output_string+'\n')

        all_predictions.extend(predictions)
        all_y_true.extend(dataset.label_indices[dataset_type][i])

    output_file.close()
    original_conll_file.close()

    new_y_pred=[]
    new_y_true=[]
    labels = dataset.unique_labels
    for i in range(len(all_y_true)):
        if labels[all_y_true[i]] != 'O' or labels[all_predictions[i]] != 'O':
            new_y_pred.append(all_predictions[i])
            new_y_true.append(all_y_true[i])
    print(sklearn.metrics.classification_report(new_y_true, new_y_pred, digits=4, #labels=dataset.index_to_label,
                                                 target_names=labels))
    return all_predictions, all_y_true, output_filepath


def predict_labels_lite(sess, model, transition_params_trained, dataset, epoch_number, stats_graph_folder, dataset_filepaths,tagging_format,main_evaluation_mode,use_crf=True):
    # Predict labels using trained model
    y_pred = {}
    y_true = {}
    output_filepaths = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue
        prediction_output = prediction_step_lite(sess, dataset, dataset_type, model, transition_params_trained, stats_graph_folder, epoch_number, dataset_filepaths,tagging_format=tagging_format,
                                            main_evaluation_mode=main_evaluation_mode,use_crf=use_crf)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output
    return y_pred, y_true, output_filepaths


