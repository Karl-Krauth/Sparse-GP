import cPickle

import savigp.full_gaussian_process


if __name__ == '__main__':
    filename = "analysis/results/seismic_21-Aug-2018_16h47m44s_135476/model.dump"
    with open(filename, 'rb') as model_file:
        model = cPickle.load(model_file)
    print "Model loaded"



