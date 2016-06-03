# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

import numpy


def linreg(inputs, targets):
    inputs = numpy.concatenate((inputs, -numpy.ones((numpy.shape(inputs)[0], 1))), axis=1)
    print(repr(inputs))
    beta = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(inputs), inputs)), numpy.transpose(inputs)),
                     targets)

    outputs = numpy.dot(inputs, beta)
    # print shape(beta)
    # print outputs
    return beta
