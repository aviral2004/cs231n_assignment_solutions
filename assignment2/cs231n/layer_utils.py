from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """Convenience layer that performs an affine transform followed be a relu nonlinearity and then a batch normalisation.
    affine -> relu -> batchnorm

    Inputs:
    - x: input to the affine layer
    - w, b: weights of the affine layer
    - gamma, beta: scale and shift params of the batchnorm layer
    - bn_param: running dict to store values for the batchnorm layer

    Output:
    - norm_scores: output from the batch_norm layer
    - cache: objects to give to the backward pass

    """
    a, fc_cache = affine_forward(x, w, b)
    norm_a, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(norm_a)
    return out, (fc_cache, bn_cache, relu_cache)

def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    dout1 = relu_backward(dout, relu_cache)
    dout2, dgamma, dbeta = batchnorm_backward_alt(dout1, bn_cache)
    dout3, dw, db = affine_backward(dout2, fc_cache)

    return dout3, dw, db, dgamma, dbeta

def affine_ln_relu_forward(x, w, b, gamma, beta, ln_param):
    a, fc_cache = affine_forward(x, w, b)
    norm_a, ln_cache = layernorm_forward(a, gamma, beta, ln_param)
    out, relu_cache = relu_forward(norm_a)
    return out, (fc_cache, ln_cache, relu_cache)

def affine_ln_relu_backward(dout, cache):
    fc_cache, ln_cache, relu_cache = cache
    dout1 = relu_backward(dout, relu_cache)
    dout2, dgamma, dbeta = layernorm_backward(dout1, ln_cache)
    dout3, dw, db = affine_backward(dout2, fc_cache)

    return dout3, dw, db, dgamma, dbeta


def gen_affine_relu_forward(x, w, b, use_dropout, dropout_param):
    dropout_cache = None

    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    if use_dropout:
        out, dropout_cache = dropout_forward(out, dropout_param)
    
    cache = (fc_cache, relu_cache, dropout_cache)
    return out, cache


def gen_affine_relu_backward(dout, cache, use_dropout):
    fc_cache, relu_cache, dropout_cache = cache

    if use_dropout:
        dout = dropout_backward(dout, dropout_cache)

    dout = relu_backward(dout, relu_cache)

    dx, dw, db = affine_backward(dout, fc_cache)

    return dx, dw, db


def gen_affine_norm_forward(x, w, b, gamma, beta, normalisation, use_dropout, bn_param, dropout_param):
    norm_cache, dropout_cache = None, None

    a, fc_cache = affine_forward(x, w, b)
    if normalisation == 'batchnorm':
        a, norm_cache = batchnorm_forward(a, gamma, beta, bn_param)
    elif normalisation == 'layernorm':
        a, norm_cache = layernorm_forward(a, gamma, beta, bn_param)
    
    out, relu_cache = relu_forward(a)
    if use_dropout:
        out, dropout_cache = dropout_forward(out, dropout_param)

    cache = (fc_cache, norm_cache, relu_cache, dropout_cache)
    return out, cache


def gen_affine_norm_backward(dout, cache, normalisation, use_dropout):
    fc_cache, norm_cache, relu_cache, dropout_cache = cache

    if use_dropout:
        dout = dropout_backward(dout, dropout_cache)

    dout = relu_backward(dout, relu_cache)

    if normalisation == 'batchnorm':
        dout, dgamma, dbeta = batchnorm_backward_alt(dout, norm_cache)
    elif normalisation == 'layernorm':
        dout, dgamma, dbeta = layernorm_backward(dout, norm_cache)

    dx, dw, db = affine_backward(dout, fc_cache)

    return dx, dw, db, dgamma, dbeta

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
