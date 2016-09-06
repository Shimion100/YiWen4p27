from __future__ import print_function;

import numpy;
import os;
import sys;
import theano
import theano.tensor as T;
import timeit

import YiWenData;
from LeNetConvPoolLayer import LeNetConvPoolLayer;
from LogisticRegression import  LogisticRegression;
from mlp import HiddenLayer


class CnnModel(object):

    def __init__(self):
        print("A void init")

    def oriGinalInit(self, batch_size):
        learning_rate = 0.05
        self.n_epochs = 50,
        self.nkerns = [20, 50]
        self.batch_size = batch_size


        """
            create Model
        """
        self.rng = numpy.random.RandomState(23455)
        dataset = 'mnist.pkl.gz'
        datasets = YiWenData.load_data(dataset)

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
            create Model
        """

        datasets = YiWenData.load_data(dataset)

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