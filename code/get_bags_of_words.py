import cv2
import numpy as np
from numpy import linalg

from distance import pdist
from feature_extraction import feature_extraction

import matplotlib.pyplot as plt

def get_bags_of_words(image_paths, feature):
    """
    This function assumes that 'vocab.mat' exists and contains an N x feature vector
    length matrix 'vocab' where each row is a kmeans centroid or visual word. This
    matrix is saved to disk rather than passed in a parameter to avoid recomputing
    the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size') below.
    """
    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    vocab_size = vocab.shape[0]
    bins = range(-1,vocab_size)

    # Your code here. You should also change the return value.
    all_histograms = np.empty((0, vocab_size))

    for path in image_paths:
        img = cv2.imread(path)[:, :, ::-1]  # 이미지 읽기

        features = feature_extraction(img, feature)  # 이미지에서 feature 추출
        dist =  pdist(vocab, features)
        min_dist_index = dist.argmin(axis=0)
        hist, _ = np.histogram(min_dist_index, bins=bins, density=True)
        #bins to check
        #plt.plot(bins[1:], hist)
        #plt.show()
        all_histograms = np.vstack((all_histograms, hist))

    return all_histograms
