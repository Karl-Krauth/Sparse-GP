import csv
import os

from matplotlib.pyplot import show, ion, savefig
import pandas
from pandas.util.testing import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt

from likelihood import SoftmaxLL, LogisticLL, UnivariateGaussian, LogGaussianCox, WarpLL, CogLL
import model_logging
from util import check_dir_exists


class PlotOutput:
    """
    A class for plotting and exporting predictions for various sorts of likelihoods.
    """

    def __init__(self):
        pass

    @staticmethod
    def plot_output(name, infile_path, model_names, filter):
        """
        Reads predictions from csv files and generates plots and output csv. Input csv files should be in the
        infile_path with following structure:

        ``infile_path`` /
                    ../any_name/
                                ../config.csv, test_.csv,train_.csv
                    ../any_name2
                                ../config.csv, test_.csv,train_.csv

        The function also exports the data used to generate graphs as csv files the following folder:
            ../graph_data
        these csv files can be used to reproduce outputs.

        Parameters
        ----------
        name : string
         name of the csv files to which data will be exported

        infile_path : string
         the folder which contains csv for configs and test and train

        model_names : list
         name of the sub-directories in ``infile_path`` to consider

        filter : callable
         a filter which will be applied in config files to filter which configs should be considered. For example,
         lambda x: x['method'] == 'full' will only consider outputs which used 'full' method
        """
        graphs = {
            'SSE': {},
            'MSSE': {},
            'NLPD': {},
            'ER': {},
            'intensity': {},
        }
        graph_n = {}
        for m in model_names:
            data_config = PlotOutput.read_config(infile_path + m + '/' + model_logging.CONFIG_FILE_NAME)
            if filter is None or filter(data_config):
                data_test = pandas.read_csv(infile_path + m + '/' + model_logging.PREDICTIONS_FILE_NAME)
                cols = data_test.columns
                dim = 0
                for element in cols:
                    if element.startswith('true_Y'):
                        dim += 1

                data_train = pandas.read_csv(infile_path + m + '/' + model_logging.TRAINING_FILE_NAME)
                Y_mean = data_train['Y_0'].mean()

                Ypred = np.array([data_test['predicted_Y_%d' % (d)] for d in range(dim)])
                Ytrue = np.array([data_test['true_Y_%d' % (d)] for d in range(dim)])
                Yvar = np.array([data_test['predicted_variance_%d' % (d)] for d in range(dim)])

                if not (PlotOutput.config_to_str(data_config) in graph_n.keys()):
                    graph_n[PlotOutput.config_to_str(data_config)] = 0
                graph_n[PlotOutput.config_to_str(data_config)] += 1

                if data_config['ll'] in [CogLL.__name__]:
                    for i in range(Ytrue.shape[0]):
                        Y_mean = data_train['Y_' + str(i)].mean()
                        PlotOutput.add_to_list(graphs['MSSE'], PlotOutput.config_to_str(data_config) + '_' + str(i),
                                               ((Ypred[i] - Ytrue[i])**2).mean() / ((Y_mean - Ytrue[i]) ** 2).mean())
                        NLPD = np.array(data_test['NLPD_' + str(i)])
                        PlotOutput.add_to_list(graphs['NLPD'], PlotOutput.config_to_str(data_config) + '_' + str(i), NLPD)

                if data_config['ll'] in [UnivariateGaussian.__name__, WarpLL.__name__]:
                    NLPD = np.array(data_test['NLPD_0'])
                    PlotOutput.add_to_list(graphs['SSE'], PlotOutput.config_to_str(data_config),
                                           (Ypred[0] - Ytrue[0])**2 / ((Y_mean - Ytrue[0]) **2).mean())
                    PlotOutput.add_to_list(graphs['NLPD'], PlotOutput.config_to_str(data_config), NLPD)

                if data_config['ll'] in [LogisticLL.__name__]:
                    NLPD = np.array(data_test['NLPD_0'])
                    PlotOutput.add_to_list(graphs['ER'], PlotOutput.config_to_str(data_config), np.array([(((Ypred[0] > 0.5) & (Ytrue[0] == -1))
                                                                 | ((Ypred[0] < 0.5) & (Ytrue[0] == 1))
                                                                 ).mean()]))
                    PlotOutput.add_to_list(graphs['NLPD'], PlotOutput.config_to_str(data_config), NLPD)

                if data_config['ll'] in [SoftmaxLL.__name__]:
                    NLPD = np.array(data_test['NLPD_0'])
                    PlotOutput.add_to_list(graphs['ER'], PlotOutput.config_to_str(data_config), np.array(
                        [(np.argmax(Ytrue, axis=0) != np.argmax(Ypred, axis=0)).mean()]))
                    PlotOutput.add_to_list(graphs['NLPD'], PlotOutput.config_to_str(data_config), NLPD)

                if data_config['ll'] in [LogGaussianCox.__name__]:
                    X0 = np.array([data_test['X_0']])

                    PlotOutput.add_to_list(graphs['intensity'], PlotOutput.config_to_str(data_config),
                                           np.array([X0[0,:]/365+1851.2026, Ypred[0, :], Yvar[0, :], Ytrue[0, :]]).T)

        for n, g in graphs.iteritems():
            if g:
                ion()
                for k in g.keys():
                    if k in graph_n.keys():
                        print k, 'n: ', graph_n[k]
                if n in ['SSE', 'NLPD']:
                    g= DataFrame(dict([(k,Series(v)) for k,v in g.iteritems()]))
                    ax = g.plot(kind='box', title=n)
                    check_dir_exists('../graph_data/')
                    g.to_csv('../graph_data/' + name  + '_' + n + '_data.csv')
                if n in ['ER', 'MSSE']:
                    g= DataFrame(dict([(k,Series(v)) for k,v in g.iteritems()]))
                    check_dir_exists('../graph_data/')
                    g.to_csv('../graph_data/' + name  + '_' + n + '_data.csv')
                    m = g.mean()
                    errors = g.std()
                    ax =m.plot(kind='bar', yerr=errors, title=n)
                    patches, labels = ax.get_legend_handles_labels()
                    ax.legend(patches, labels, loc='lower center')
                if n in ['intensity']:
                    X = g.values()[0][:, 0]
                    true_data = DataFrame({'x': X, 'y': g.values()[0][:, 3]})
                    true_data.to_csv('../graph_data/' + name  + '_' + 'true_y' + '_data.csv')
                    plt.figure()
                    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
                    c = 0
                    check_dir_exists('../graph_data/')
                    graph_data = DataFrame()
                    for k,v in g.iteritems():
                        # plt.plot(X, v[:, 1], hold=True, color=color[c], label=k)
                        # plt.fill_between(X, v[:, 1] - 2 * np.sqrt(v[:, 2]), v[:, 1] + 2 * np.sqrt(v[:, 2]), alpha=0.2, facecolor=color[c])
                        graph_data = graph_data.append(DataFrame({'x': X, 'm' : v[:, 1], 'v' :v[:, 2], 'model_sp' :[k] * X.shape[0]}
                                                                 ))
                        c += 1
                    plt.legend(loc='upper center')
                    graph_data.to_csv('../graph_data/' + name  + '_' + n + '_data.csv')

                show(block=True)

    @staticmethod
    def add_to_list(l, name, value):
        if not (name in l):
            l[name] = value
        else:
            l[name] = np.hstack((l[name], value))

    @staticmethod
    def config_to_str(config):
        return config['method'] +':' + str(config['sparsity_factor'])

    @staticmethod
    def plot_output_all(name, path, filter):
        """
        A helper function which will extract plot data for all the folders in ``path`` which satisfy the filter.

        Parameters
        ----------
        name : string
         name of the folder used to export csv files

        path : string
         path which contains folders to plot

        filter : callable
         a filter which will be applied in config files to filter which configs should be considered. For example,
         lambda x: x['method'] == 'full' will only consider outputs which used 'full' method
        """
        dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d[0] != '.']
        PlotOutput.plot_output(name, path, dir, filter)


    @staticmethod
    def find_all(path, filter):
        """
        Prints the name of all folders inside ``path`` which their config files satisfies ``filter``
        """
        dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for m in dir:
            data_config = PlotOutput.read_config(path + m + '/' + model_logging.CONFIG_FILE_NAME)
            if filter is None or filter(data_config):
                print m

    @staticmethod
    def read_config(path):
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            return reader.next()

if __name__ == '__main__':
    PlotOutput.plot_output_all('boston', model_logging.OUTPUT_PATH,
                               lambda x: x['experiment'] == 'boston')
