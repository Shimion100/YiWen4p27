# coding=utf-8
from __future__ import print_function

import gzip
import numpy
import os
import six.moves.cPickle as pickle
import sys
import theano
import theano.tensor as T
import timeit
from PIL import Image
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

import CnnModel;
from mlp import HiddenLayer

"""
    这个肯爹货,竟然把他写成函数了,没有作成类,fuck!
"""

def predict():
    print("Load model.....")

    # load the saved model
    cnnModel = pickle.load(open('best_model.pkl', 'rb'))
    aIndex = T.lscalar()
    # compile a predictor function
    # the parameter is works;

    predict_model = theano.function(
        inputs=[aIndex],
        outputs=cnnModel.errorsPred(2))

    predicted_values = predict_model()
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)



"""
    主方法
"""
if __name__ == '__main__':
    #"""
    model = CnnModel()
    model.oriGinalInit(120)
    model.trainModel();
    model.modifyModel(10, 'mnist.pkl.gz')
    # save the best model
    print("Start to save the Model!")
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    """
    predict()
    """
