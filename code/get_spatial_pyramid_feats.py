import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction
from sklearn.preprocessing import normalize


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_hog.npy' (for HoG) or 'vocab_sift.npy' (for SIFT)
    exists and contains an N x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """
    def _get_histogram_for_feature(img, vocab, feature, bins):
        features = feature_extraction(img, feature)
        try:
            dist = pdist(vocab, features)
            min_dist_index = dist.argmin(axis=0)
            hist, _ = np.histogram(min_dist_index, bins=bins)
            return hist
        except:
            hist, _ = np.histogram([], bins=bins)
            return hist

    def _spatial_pyramid_recursion(img, max_level, current_level, vocab, feature):
        if current_level > max_level:
            return np.zeros(vocab.shape[0])
        else:
            img1 = img[0:int(img.shape[0]/2), 0:int(img.shape[1]/2),:]
            img2 = img[0:int(img.shape[0]/2), int(img.shape[1] / 2):int(img.shape[1]), :]
            img3 = img[int(img.shape[0]/2):int(img.shape[0]), 0:int(img.shape[1]/2),:]
            img4 = img[int(img.shape[0]/2):int(img.shape[0]), int(img.shape[1] / 2):int(img.shape[1]), :]

            #NOTE: visual check of division
            #cv2.imshow('img1',img1)
            #cv2.imshow('img2', img2)
            #cv2.imshow('img3', img3)
            #cv2.imshow('img4', img4)

            hist_img = _get_histogram_for_feature(img, vocab, feature, bins)

            current_weight = pow(2,current_level-max_level)

            all_histograms = np.array([current_weight*hist_img,
                          _spatial_pyramid_recursion(img1, max_level, current_level + 1, vocab, feature),
                          _spatial_pyramid_recursion(img2, max_level, current_level + 1, vocab, feature),
                          _spatial_pyramid_recursion(img3, max_level, current_level + 1, vocab, feature),
                          _spatial_pyramid_recursion(img4, max_level, current_level + 1, vocab, feature)])

            return all_histograms.sum(axis=0)

    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.

    bins = range(-1,vocab_size)

    # Your code here. You should also change the return value.
    all_histograms = np.empty((0, vocab_size))

    i = 0
    for path in image_paths:
        img = cv2.imread(path)[:, :, ::-1]
        print('iter: ' + str(i))
        sp_histograms = _spatial_pyramid_recursion(img, max_level, 1, vocab, feature)
        sp_histograms = normalize(sp_histograms.reshape(1, -1), norm="l2")

        all_histograms = np.vstack((all_histograms, sp_histograms))
        i += 1

    return all_histograms
    #return np.zeros((1500, 36))
