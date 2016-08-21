# encoding: utf-8

import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import warnings
import time
warnings.filterwarnings("ignore")
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

theano.config.floatX = "float32"

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def predict_conv_net(testdata, U, img_w = 100, filter_hs = [3, 4, 5], hidden_units = [50, 8], dropout_rate = [0.5], batch_size = 100,
                     test_batch_size=500, conv_non_linear = "relu", activations = [Iden],  model_params = []):

    rng = np.random.RandomState(3435)
    img_h = len(testdata[0])
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    paraNum = 0
    for i in xrange(len(filter_hs)):
        paraNum += 2
        W_conv_values = model_params[2 * i].get_value()
        W_conv = theano.shared(value=W_conv_values, borrow=True, name="W_conv")
        b_conv_values = model_params[2 * i + 1].get_value()
        b_conv = theano.shared(value=b_conv_values, borrow=True, name="b_conv")
        # print W_conv, b_conv
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear,  W=W_conv, b=b_conv)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)
    W_values = model_params[paraNum].get_value()
    W = theano.shared(value=W_values, borrow=True, name="W")
    b_values = model_params[paraNum + 1].get_value()
    b = theano.shared(value=b_values, borrow=True, name="b")
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate, logW=W, logb=b)


    # for conv_layer in conv_layers:
    #     print conv_layerlogW.params[0].get_value(), conv_layer.params[1].get_value()
    # print classifier.params[0].get_value(), classifier.params[0].get_value()
    # for word in Words.get_value():
    #     print word

    test_set_x = testdata
    # test_pred_layers = []
    # test_size = test_set_x.shape[0]

    # test_layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((test_size, 1, img_h, Words.shape[1]))
    # for conv_layer in conv_layers:
    #     test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
    #     test_pred_layers.append(test_layer0_output.flatten(2))
    # test_layer1_input = T.concatenate(test_pred_layers, 1)
    # test_y_pred = classifier.predict(test_layer1_input)
    # test_model_all = theano.function([x], test_y_pred, allow_input_downcast=True)

    test_pred_layers_a = []
    test_pred_layers_b = []

    test_size = test_set_x.shape[0]
    test_iter = int(test_size / test_batch_size)
    extra_test_size = test_size - test_iter * test_batch_size

    test_layer0_input_a = Words[T.cast(x.flatten(), dtype="int32")].reshape((test_batch_size, 1, img_h, Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output_a = conv_layer.predict(test_layer0_input_a, test_batch_size)
        test_pred_layers_a.append(test_layer0_output_a.flatten(2))
    test_layer1_input_a = T.concatenate(test_pred_layers_a, 1)
    test_y_pred_a = classifier.predict(test_layer1_input_a)

    test_layer0_input_b = Words[T.cast(x.flatten(), dtype="int32")].reshape((extra_test_size, 1, img_h, Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output_b = conv_layer.predict(test_layer0_input_b, extra_test_size)
        test_pred_layers_b.append(test_layer0_output_b.flatten(2))
    test_layer1_input_b = T.concatenate(test_pred_layers_b, 1)
    test_y_pred_b = classifier.predict(test_layer1_input_b)

    test_model_all_a = theano.function([x], test_y_pred_a, allow_input_downcast=True)
    test_model_all_b = theano.function([x], test_y_pred_b, allow_input_downcast=True)

    print 'predicting ...'
    prediction1 = np.zeros((test_iter, test_batch_size))
    for j in xrange(test_iter):
        prediction1[j] = test_model_all_a(test_set_x[test_batch_size * j:test_batch_size * (j + 1), :])

    prediction2 = test_model_all_b(test_set_x[-extra_test_size:, :])
    prediction = list(prediction1.reshape(test_batch_size * test_iter)) + list(prediction2)
    return prediction
    
def get_idx_from_sent(sent, word_idx_map, max_l=500, k=100, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.  (zero-padding)
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    count = 0
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
            count += 1
            if count >= max_l:
                break
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x

#将文本分为训练集和验证集, 其中9个作为训练集合, 1个作为验证集合
def make_idx_data_cv(revs, word_idx_map, max_l=500, k=100, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    test = []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        test.append(sent)
    test = np.array(test, dtype="int")
    return test
  
   
if __name__=="__main__":
    filePath = "locationTest.p"
    print "loading data...",
    x = cPickle.load(open(filePath, "r"))
    revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
    print "data loaded!"

    # [W_conv, b_conv, W_conv, b_conv, W_conv, b_conv, W, b]
    print "loading params...",
    with open('./locationModel', 'r') as f:
        model_params = cPickle.load(f)
    print "params loaded!"
    # print model_params[0].get_value()

    execfile("conv_net_classes.py")
    print "using: word2vec vectors"

    # y_test = np.array(list(np.ones(100)) + list(np.zeros(100)))  # the actual test classes
    # testdata = make_idx_data_cv(revs, word_idx_map, max_l=500, k=100, filter_h=5)
    # y_prediction = predict_conv_net(testdata, W, filter_hs=[3,4,5], conv_non_linear="relu", hidden_units=[50,8],batch_size=100,
    #                                 dropout_rate=[0.5], model_params = model_params)
    # acc = np.sum(y_test == y_prediction, axis=0) * 100 / float(len(y_test))
    # print 'Test Accuracy' + ' : ' + str(acc) + ' %'
    # print y_prediction
    testdata = make_idx_data_cv(revs, word_idx_map, max_l=500, k=100, filter_h=5)
    y_prediction = predict_conv_net(testdata, W, filter_hs=[3, 4, 5], conv_non_linear="relu", hidden_units=[50, 8],
                                     test_batch_size=500, batch_size=100, dropout_rate=[0.5], model_params=model_params)
    with open('./locationPredict', 'w') as f:
        for index in xrange(0, len(y_prediction)):
            # if y_prediction[index] == 7:
            #     f.write(revs[index]["uid"] + ',东北' + '\n')
            # elif y_prediction[index] == 6:
            #     f.write(revs[index]["uid"] + ',华北' + '\n')
            # elif y_prediction[index] == 5:
            #     f.write(revs[index]["uid"] + ',华东' + '\n')
            # elif y_prediction[index] == 4:
            #     f.write(revs[index]["uid"] + ',华中' + '\n')
            # elif y_prediction[index] == 3:
            #     f.write(revs[index]["uid"] + ',华南' + '\n')
            # elif y_prediction[index] == 2:
            #     f.write(revs[index]["uid"] + ',西南' + '\n')
            # elif y_prediction[index] == 1:
            #     f.write(revs[index]["uid"] + ',西北' + '\n')
            # else:
            #     f.write(revs[index]["uid"] + ',境外' + '\n')
            f.write(revs[index]["uid"] + ' ' + str(y_prediction[index]) + '\n')




