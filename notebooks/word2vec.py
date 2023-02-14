
# %% imports
import numpy as np
import random
from utils.utils import *

# %% basic functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def naiveSoftmaxLossAndGradient(
    centerWordVec, # (word_vector_dim,)
    outsideWordIdx,# (index of outside word)
    outsideVectors,# (vocab_size, word_vector_dim)
    dataset
):
    softmax = lambda x: np.exp(x) / (np.sum(np.exp(x) + 1e-14))
    y_hat = softmax(outsideVectors @ centerWordVec)
    loss = -np.log(y_hat[outsideWordIdx])

    gradCenterVec = outsideVectors.T @ y_hat
    gradOutsideVecs = np.outer(y_hat, centerWordVec)

    return loss, gradCenterVec, gradOutsideVecs

def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
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

    negativeSamplesIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negativeSamplesIndices

    # (1, word_vector_dim).T @ (word_vector_dim, 1) = (1, vocab_size)
    dot_1 = outsideVectors[outsideWordIdx].T @ centerWordVec
    dot_2 = -outsideVectors[negativeSamplesIndices] @ centerWordVec
    loss = -np.log(sigmoid(dot_1)) - np.sum(np.log(sigmoid(dot_2)))

    gradCenterVec = (sigmoid(dot_1) - 1) * outsideVectors[outsideWordIdx]

    gradOutsideVecs = np.zeros(outsideVectors.shape)
    gradOutsideVecs[outsideWordIdx] = (sigmoid(dot_1) - 1) * centerWordVec

    for i, negativeSample in enumerate(negativeSamplesIndices):
        gradOutsideVecs[negativeSample] += (sigmoid(dot_2[i]) - 1) * centerWordVec

    return loss, gradCenterVec, gradOutsideVecs



#%% run
dataset, dummy_vectors, dummy_tokens = getDummyObjects()
#naiveSoftmaxLossAndGradient(np.random.randn(3), 1, dummy_vectors, dataset)
negSamplingLossAndGradient(np.random.randn(3), 1, dummy_vectors, dataset)

# %%
