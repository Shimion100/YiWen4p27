# coding=utf-8
from __future__ import print_function
from PIL import Image
import gzip
import os
import sys
import timeit
import six.moves.cPickle as pickle
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from mlp import HiddenLayer



"""
    贾晓栋,这个是用来作加载数据的.
"""
def load_data(dataset):

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

"""
    逻辑回归的类,也是卷积神经网络的最后一层.
"""

class LogisticRegression(object):


    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

"""
    这个是主要卷积神经网络的代码
"""

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def modify(self, input, image_shape):
        print("Modifying----------------------")
        self.input = input
        self.image_shape = image_shape


    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):


        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

"""
    这个肯爹货,竟然把他写成函数了,没有作成类,fuck!
"""

class CnnModel(object):

    def __init__(self):
        print("A void init")

    def oriGinalInit(self, batch_size):
        learning_rate = 0.05
        self.n_epochs = 50,

        self.nkerns = [20, 50]
        self.batch_size = batch_size


        """
            创建Model
        """
        self.rng = numpy.random.RandomState(23455)
        dataset = 'mnist.pkl.gz'
        datasets = load_data(dataset)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches //= self.batch_size
        self.n_valid_batches //= self.batch_size
        self.n_test_batches //= self.batch_size


        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        aIndex = T.lscalar()
        # start-snippet-1
        x = T.matrix('x')  # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
        # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # Reshape matrix of rasterized images of shape (self.batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        self.layer0_input = x.reshape((self.batch_size, 1, 28, 28))


        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (self.batch_size, self.nkerns[0], 12, 12)
        self.layer0 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer0_input,
            image_shape=(self.batch_size, 1, 28, 28),
            filter_shape=(self.nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (self.batch_size, self.nkerns[1], 4, 4)
        self.layer1 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer0.output,
            image_shape=(self.batch_size, self.nkerns[0], 12, 12),
            filter_shape=(self.nkerns[1], self.nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (self.batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (self.batch_size, self.nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer2_input = self.layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            self.rng,
            input=self.layer2_input,
            n_in=self.nkerns[1] * 4 * 4,
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=7)

        # the cost we minimize during training is the NLL of the model
        cost = self.layer3.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [index],
            self.layer3.errors(y),
            givens={
                x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        self.errorsPred = theano.function(
            inputs=[aIndex],
            outputs=self.layer3.y_pred,
            givens={
                x: test_set_x[aIndex * 10: (aIndex + 1) * 10],
            }
        )
        self.validate_model = theano.function(
            [index],
            self.layer3.errors(y),
            givens={
                x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # self.train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        self.train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

    def trainModel(self,):
        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        # early-stopping parameters
        patience = 100  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        validation_frequency = min(self.n_train_batches, patience // 2)
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):

            epoch = epoch + 1
            for minibatch_index in range(self.n_train_batches):

                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = self.train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in range(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in range(self.n_test_batches)
                            ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    def modifyModel(self, batch_size, dataset):
        print("Start to modify the model---------------------------")
        dataset = dataset
        self.batch_size = batch_size

        """
            创建Model
        """

        datasets = load_data(dataset)

        test_set_x, test_set_y = datasets[2]

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # start-snippet-1
        x = T.matrix('x')  # the data is presented as rasterized images

        # [int] labels

        print('... building the model')

        self.layer0_input = x.reshape((self.batch_size, 1, 28, 28))

        self.layer0.modify(self.layer0_input, (self.batch_size, 1, 28, 28))

        self.layer1.modify(self.layer0.output, (self.batch_size, self.nkerns[0], 12, 12))

        self.layer2_input = self.layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer2 = self.layer2

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=7)

        print("-------batch_size---------batch_size---------batch_size------------",self.layer0.image_shape)


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
