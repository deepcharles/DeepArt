__author__ = 'charles'

import theano.tensor as T


def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g


def content_loss(P, X, layer):
    p = P[layer]
    x = X[layer]

    loss = 1. / 2 * ((x - p) ** 2).sum()
    return loss


def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)

    N = a.shape[1]
    M = a.shape[2] * a.shape[3]

    loss = 1. / (4 * N ** 2 * M ** 2) * ((G - A) ** 2).sum()
    return loss


def total_variation_loss(x):
    return (((x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 + (
    x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2) ** 1.25).sum()