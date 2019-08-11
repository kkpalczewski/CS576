import numpy as np


def get_features_from_pca(feat_num, feature):
    """
    This function loads 'vocab_sift.npy' or 'vocab_hog.npg' file and
    returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
    :param feature: 'Hog' or 'SIFT'

    :return: an N x feat_num matrix
    """

    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    # Your code here. You should also change the return value.

    def _get_PCA_vectors(feat_num, vocab):

        mean = vocab.mean(axis=0, keepdims=True)
        vocab_normalized = vocab - np.multiply(np.ones([vocab.shape[0], mean.shape[0]]),
                                           mean)
        #TEST: mean unit test
        #mean = vocab_normalized.mean(axis=0, keepdims=True)

        cov_matrix = np.cov(np.transpose(vocab_normalized))
        sigma, V = np.linalg.eig(cov_matrix)
        order_sigma = np.argsort(sigma)

        PCA_vectors = []
        i = 1
        for f in range(len(order_sigma)):
            eigen_vector = V[:, order_sigma[i]]
            if all(True for _ in np.isreal(eigen_vector)):
                PCA_vectors.append(np.real(eigen_vector))
                i += 1
            if len(PCA_vectors) == feat_num:
                break

        return np.array(PCA_vectors)

    #MAIN
    PCA_vectors = _get_PCA_vectors(feat_num, vocab)

    d = np.dot(vocab, np.transpose(PCA_vectors))

    return np.dot(vocab, np.transpose(PCA_vectors))
    #return np.zeros((vocab.shape[0],2))

