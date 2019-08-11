import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats: an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels: an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats: an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type: SVM kernel type. 'linear' or 'RBF'

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)

    categories_dict = dict(zip(np.arange(15), categories))

    all_predicted_proba = np.empty([1500,1])

    for c in range(len(categories)):
        one_vs_all_labels = [1 if n == categories[c] else -1 for n in train_labels]
        clf = svm.SVC(probability=True, gamma='auto', kernel=kernel_type)
        clf.fit(train_image_feats, one_vs_all_labels)
        predicted_proba = clf.predict_proba(test_image_feats)
        predicted_proba = predicted_proba[:,1].reshape((len(predicted_proba[:,1]),1))
        all_predicted_proba = np.hstack((all_predicted_proba,
                                                          predicted_proba))

    all_predicted_proba = all_predicted_proba[:,1:]

    max_proba = np.argmax(all_predicted_proba, axis=1)

    out_labels = np.array([categories_dict[x] for x in max_proba])

    return out_labels