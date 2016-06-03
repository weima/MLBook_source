# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

import numpy


class Pcn:
    """ A basic Perceptron (the same pcn.py except with the weights printed
    and it does not reorder the inputs)"""

    def __init__(self, inputs, targets):
        """ Constructor """
        # Set up network size
        self.outputs = self.pcn_fwd(inputs)
        if numpy.ndim(inputs) > 1:
            self.nIn = numpy.shape(inputs)[1]
        else:
            self.nIn = 1

        if numpy.ndim(targets) > 1:
            self.nOut = numpy.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = numpy.shape(inputs)[0]

        # Initialise network
        self.weights = numpy.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05

    def pcn_train(self, inputs, targets, eta, n_iterations):
        """ Train the thing """
        # Add the inputs that match the bias node
        inputs = numpy.concatenate((inputs, -numpy.ones((self.nData, 1))), axis=1)

        # Training
        change = range(self.nData)

        for n in range(n_iterations):
            self.weights += eta * numpy.dot(numpy.transpose(inputs), targets - self.outputs)
            print("Iteration: ", n)
            print(self.weights)

            activations = self.pcn_fwd(inputs)
            print("Final outputs are:")
            print(activations)
            # return self.weights

    def pcn_fwd(self, inputs):
        """ Run the network forward """

        outputs = numpy.dot(inputs, self.weights)

        # Threshold the outputs
        return numpy.where(outputs > 0, 1, 0)

    def confmat(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = numpy.concatenate((inputs, -numpy.ones((self.nData, 1))), axis=1)
        outputs = numpy.dot(inputs, self.weights)

        nClasses = numpy.shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = numpy.where(outputs > 0, 1, 0)
        else:
            # 1-of-N encoding
            outputs = numpy.argmax(outputs, 1)
            targets = numpy.argmax(targets, 1)

        cm = numpy.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i, j] = numpy.sum(numpy.where(outputs == i, 1, 0) * numpy.where(targets == j, 1, 0))

        print(cm)
        print(numpy.trace(cm) / numpy.sum(cm))
