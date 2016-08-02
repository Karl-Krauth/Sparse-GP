import cPickle
import sys
import data_source 
import data_transformation

directory = sys.argv[1]
data = data_source.airline_data()[0]
test_inputs = data["test_inputs"]
test_outputs = data["test_outputs"]
train_inputs = data["train_inputs"]
train_outputs = data["train_outputs"]

transformer = data_transformation.MeanStdTransformation(train_inputs, train_outputs)
test_inputs = transformer.transform_X(test_inputs)
test_outputs = transformer.transform_Y(test_outputs)

for i in xrange(200):
    model_image_file_path = directory + str(i)
    with open(model_image_file_path) as model_image_file:
        model = cPickle.load(model_image_file)
    mu, var, nlpd = model.predict(test_inputs, test_outputs)
    nlpd = transformer.untransform_NLPD(nlpd)[:, 0]
    print "1", max(nlpd), nlpd.mean(), nlpd.std()
