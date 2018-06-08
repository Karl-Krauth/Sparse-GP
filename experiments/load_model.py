import cPickle

import savigp.full_gaussian_process


if __name__ == '__main__':
    filename = "../../results/savigp-results/seismic/seismic_01-Jun-2018_13h33m59s_3234/model.dump"
    with open(filename, 'rb') as model_file:
        model = cPickle.load(model_file)




