"""This module provides facilities for logging and exporting information relevant to a model."""
import csv
import cPickle
import datetime
import logging
import os

import numpy as np

import gaussian_process
import util

OUTPUT_PATH = '../results/'
CONFIG_FILE_NAME = 'config.csv'
PREDICTIONS_FILE_NAME = 'predictions.csv'
TRAINING_FILE_NAME = 'train.csv'
MODEL_FILE_NAME = 'model.dump'
LOG_LEVEL = logging.DEBUG

_log_folder_path = None
logger = None


def init_logger(name, output_to_disk=True):
    """
    Initialize the logger and the information necessary to save the model.

    Parameters
    ----------
    name : str
        The name of the experiment currently being run.
    """
    global logger
    global _log_folder_path

    # Configure the logger.
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if output_to_disk:
        # Add a file handler to log to disk.
        log_start_time = datetime.datetime.now()
        folder_name = name + '_' + log_start_time.strftime('%d-%b-%Y_%Hh%Mm%Ss') + '_%d' % os.getpid()
        _log_folder_path = os.path.join(OUTPUT_PATH, folder_name)
        util.check_dir_exists(_log_folder_path)
        file_handler = logging.FileHandler(os.path.join(_log_folder_path, name + '.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add a stream handler to log to stdout.
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def disable_logger():
    logger.setLevel(logging.CRITICAL)

def export_training_data(X, Y, export_X=False):
    """
    Export the training data into a csv file.

    Parameters
    ----------
    X : ndarray
        The input training data.
    Y : ndarray
        The output training data.
    export_X : boolean
        Whether to export 'X', if False only 'Y' will be exported.
    """
    if _log_folder_path is None:
        print "Warning: logger is not initialized."
        return

    # Generate headers and prepare the data for output.
    header = ['Y_%d' % (j) for j in xrange(Y.shape[1])]
    if export_X:
        header += ['X_%d' % (j) for j in xrange(X.shape[1])]
        data = np.hstack((Y, X))
    else:
        data = Y

    header = ','.join(header)

    # Write the data out to a csv file.
    file_path = os.path.join(_log_folder_path, TRAINING_FILE_NAME)
    np.savetxt(file_path, data, header=header, delimiter=',', comments='')


def export_predictions(X, true_Y, predicted_Y, predicted_variance, nlpd, export_X=False):
    """
    Export test data and the associated predictions into a csv file.

    Parameters
    ----------
    X : ndarray
        The test inputs for which prediction have been made.
    true_Y : ndarray
        The true output values.
    predicted_Y : ndarray
        The predicted output values.
    predicted_variance : ndarray
        Variance of the prediction.
    nlpd : ndarray
        Negative log probability density of the predictions.
    export_X : boolean
        Whether to export 'X' to the csv file. If False, only 'Y will be exported
        (useful in large datasets).
    """
    if _log_folder_path is None:
        print "Warning: logger is not initialized."
        return

    header = (
        ['true_Y_%d' % (j) for j in xrange(true_Y.shape[1])] +
        ['predicted_Y_%d' % (j) for j in xrange(predicted_Y[0].shape[1])] +
        ['predicted_variance_%d' % (j) for j in xrange(predicted_variance[0].shape[1])] +
        ['NLPD_%d' % (j) for j in xrange(nlpd.shape[1])]
    )
    data = [true_Y] + predicted_Y + predicted_variance + [nlpd]
    print true_Y.shape, nlpd.shape
    if export_X:
        header += ['X_%d' % (j) for j in xrange(X.shape[1])]
        data.append(X)

    # Collapse the header and the data.
    header = ','.join(header)
    data = np.hstack(data)

    # Write the data out to a csv file.
    file_path = os.path.join(_log_folder_path, PREDICTIONS_FILE_NAME)
    np.savetxt(file_path, data, header=header, delimiter=',', comments='')


def export_configuration(config):
    """
    Export model configuration information as well as optimisation parameters to a csv file.

    Parameters
    ----------
    config : dictionary
        Configuration to be exported.
    """
    if _log_folder_path is None:
        print "Warning: logger is not initialized."
        return

    file_path = os.path.join(_log_folder_path, CONFIG_FILE_NAME)
    with open(file_path, 'wb') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=config.keys())
        writer.writeheader()
        writer.writerow(config)


def snapshot_model(model, name=MODEL_FILE_NAME):
    """
    Save the current state of the model to be restored at a later date.

    Parameters
    ----------
    model : Model
        The model we wish to snapshot.
    """
    if _log_folder_path is None:
        return

    file_path = os.path.join(_log_folder_path, name)
    with open(file_path, 'w') as model_file:
        cPickle.dump(model, model_file, protocol=cPickle.HIGHEST_PROTOCOL)
