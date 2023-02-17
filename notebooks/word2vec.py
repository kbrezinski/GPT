
# %% imports
import numpy as np
import random
from utils.utils import *

# %% basic functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, eps=1e-9):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + eps)

def naiveSoftmaxLossAndGradient(
    centerWordVec, # (word_vector_dim,)
    outsideWordIdx,# (index of outside word)
    outsideVectors,# (vocab_size, word_vector_dim)
    dataset
):

    # (vocab_size, word_vector_dim) @ (word_vector_dim,) = (vocab_size, 1)
    y_hat = softmax(outsideVectors @ centerWordVec)
    loss = -np.log(y_hat[outsideWordIdx])

    # (vocab_size, word_vector_dim).T @ (vocab_size, 1) = (word_vector_dim, 1)
    gradCenterVec = outsideVectors.T @ y_hat
    # (vocab_size, 1) .outer (word_vector_dim, 1) = (vocab_size, word_vector_dim)
    gradOutsideVecs = np.outer(y_hat, centerWordVec)

    return loss, gradCenterVec, gradOutsideVecs

def negSamplingLossAndGradient(
    centerWordVec, # (word_vector_dim,)
    outsideWordIdx,# (index of outside word)
    outsideVectors,# (vocab_size, word_vector_dim)
    dataset,
    K=10
):
    def getNegativeSamples(outsideWordIdx, dataset, K):
        negativeSamples = []
        while len(negativeSamples) < K:
            negativeSample = dataset.sampleTokenIdx()
            if negativeSample != outsideWordIdx:
                negativeSamples.append(negativeSample)
        return negativeSamples
    
    # sample some random indices and add them as context words
    negativeSamplesIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    negativeSamplesIndices = [outsideWordIdx] + negativeSamplesIndices

    # (word_vector_dim, 1).T @ (word_vector_dim, 1) = (1, 1)
    dot_1 = outsideVectors[outsideWordIdx].T @ centerWordVec
    # (K, word_vector_dim) @ (word_vector_dim, 1) = (K, 1)
    dot_2 = -outsideVectors[negativeSamplesIndices] @ centerWordVec
    # (1,) - sum(K, 1) = (1,)
    loss = -np.log(sigmoid(dot_1)) - np.sum(np.log(sigmoid(dot_2)))

    # (1, 1) * (word_vector_dim, 1) = (word_vector_dim, 1)
    gradCenterVec = (sigmoid(dot_1) - 1) * outsideVectors[outsideWordIdx]

    # (vocab_size, word_vector_dim)
    gradOutsideVecs = np.zeros(outsideVectors.shape)

    # (1, 3) = (1,) * (3,)
    gradOutsideVecs[outsideWordIdx] = (sigmoid(dot_1) - 1) * centerWordVec

    for i, negativeSample in enumerate(negativeSamplesIndices):
        gradOutsideVecs[negativeSample] += (sigmoid(dot_2[i]) - 1) * centerWordVec

    return loss, gradCenterVec, gradOutsideVecs

def skipgram(
    currentCenterWord,
    windowSize,
    outsideWords,
    word2Ind,
    centerWordVectors,
    outsideVectors,
    dataset,
    word2vecLossAndGradient=naiveSoftmaxLossAndGradient
):
    loss = 0.0
    # (vocab_size, word_vector_dim)
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    # (vocab_size, word_vector_dim)
    gradOutsideVecs = np.zeros(outsideVectors.shape)

    # convert word to index
    centerWordIdx = word2Ind[currentCenterWord]
    # get center word vector
    centerWordVec = centerWordVectors[centerWordIdx]

    # loop over outside words
    for outsideWord in outsideWords:
        # get outside word index
        outsideWordIdx = word2Ind[outsideWord]
        # get loss and gradients using naiveSoftmaxLossAndGradient
        loss_, gradCenterVec_, gradOutsideVecs_ = word2vecLossAndGradient(
            centerWordVec,
            outsideWordIdx,
            outsideVectors,
            dataset
        )
        loss += loss_
        # (1, 1) - update center word vector for each outer word
        gradCenterVecs[centerWordIdx] += gradCenterVec_
        # (K, 3) - update outside word vectors for each outer word
        gradOutsideVecs += gradOutsideVecs_

    return loss, gradCenterVecs, gradOutsideVecs


dataset, dummy_vectors, dummy_tokens = getDummyObjects()
_ = naiveSoftmaxLossAndGradient(np.random.randn(3), 1, dummy_vectors, dataset)
_ = negSamplingLossAndGradient(np.random.randn(3), 1, dummy_vectors, dataset)
skipgram("c", 1, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors, dummy_vectors, dataset)
# %%
