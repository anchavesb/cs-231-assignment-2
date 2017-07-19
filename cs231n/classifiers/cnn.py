from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.size = 3

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        #Convolutional layer
        conv_stride = 1
        P = (filter_size - 1) / 2  #pad
        C, H, W = input_dim
        F = num_filters
        filter_height = filter_size
        filter_width = filter_size
        self.params['W1'] = np.random.randn(F,  C, filter_height, filter_width) * weight_scale
        self.params['b1'] = np.zeros(F)
        #Conv output module
        cH = (H + 2*P - filter_height) / conv_stride + 1
        cW = (W + 2*P - filter_width) / conv_stride + 1
        #(F, cH, cW)
        #Pool output
        pool_stride = 2
        pool_width = 2
        pool_height = 2
        pH = (int)((cH - pool_height) / pool_stride + 1)
        pW = (int)((cW - pool_width ) / pool_stride + 1)
        #(F, pH, pW)
        # Affine layer hidden
        # Input size: (N, F * pH * pW)
        # Output: (N, hidden_dim)
        hH = hidden_dim
        self.params['W2'] = np.random.randn(F * pH * pW, hH) * weight_scale
        self.params['b2'] = np.zeros(hH)
        # Affine layer
        # Input size: (N, hH)
        # Output size: (N, num_classes)
        nC = num_classes
        self.params['W3'] = np.random.randn(hH, nC) * weight_scale
        self.params['b3'] = np.zeros(nC)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        """
            A three-layer convolutional network with the following architecture:

            conv - relu - 2x2 max pool - affine - relu - affine - softmax

            The network operates on minibatches of data that have shape (N, C, H, W)
            consisting of N images, each with height H and width W and with C input
            channels.
        """
        out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        N, F, pH, pW = out.shape
        x = out.reshape((N, F * pH * pW))
        out, affine_relu_cache = affine_relu_forward(x, W2, b2)

        scores, cache_scores = affine_forward(out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dloss = softmax_loss(scores, y)
        #Regularization
        loss += 0.5 * self.reg * np.sum(W1 ** 2)
        loss += 0.5 * self.reg * np.sum(W2 ** 2)
        loss += 0.5 * self.reg * np.sum(W3 ** 2)
        #Gradients
        dx3, dW3, db3 = affine_backward(dloss, cache_scores)
        dW3 += self.reg * W3
        grads['W3'] = dW3
        grads['b3'] = db3

        dx2, dW2, db2 = affine_relu_backward(dx3, affine_relu_cache)
        dW2 += self.reg * W2
        grads['W2'] = dW2
        grads['b2'] = db2

        dx2 = dx2.reshape(N,  F, pH, pW)
        dx1, dW1, db1 = conv_relu_pool_backward(dx2, conv_cache)
        dW1 += self.reg * W1
        grads['W1'] = dW1
        grads['b1'] = db1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
