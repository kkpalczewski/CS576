import numpy as np
from distance import pdist
from scipy.spatial.distance import euclidean

def kmeans_clustering(all_features, vocab_size, epsilon, max_iter):
    """
    The function kmeans implements a k-means algorithm that finds the centers of vocab_size clusters
    and groups the all_features around the clusters. As an output, centroids contains a
    center of the each cluster.

    :param all_features: an N x d matrix, where d is the dimensionality of the feature representation.
    :param vocab_size: number of clusters.
    :param epsilon: When the maximum distance between previous and current centroid is less than epsilon,
        stop the iteration.
    :param max_iter: maximum iteration of the k-means algorithm.

    :return: an vocab_size x d array, where each entry is a center of the cluster.
    """

    # Your code here. You should also change the return value.

    def _initiate_random_centroids(all_features, vocab_size):
        """
        Initiate random centroids in the range of input

        :param all_features:
        :param vocab_size:
        :return:
        """
        centroids = []
        # 1) Genereate points for initial centroids

        min_feat = np.ones(all_features[0].size)*np.inf
        max_feat = np.zeros(all_features[0].size)

        for a in all_features:
            for p in range(len(a)):
                if a[p] < min_feat[p]:
                    min_feat[p] = a[p]
                else:
                    if a[p] > max_feat[p]:
                        max_feat[p] = a[p]


        for _ in range(vocab_size):
            random_vector = np.multiply(np.random.rand(1, all_features[0].size),
                                    max_feat-min_feat) + min_feat
            centroids.append(random_vector.flatten())

        return np.array(centroids)

    def _assign_vectors_to_nearest_centroid(all_features, centroid):
        """
        Assign vectors to nearest centroids

        :param all_features:
        :param centroid:
        :return:
        """
        #TODO: sprawdz co lepiej dziala
        new_centroid_coor = np.zeros([len(centroid), all_features[0].size])
        #new_centroid_coor = centroid
        new_centroid_counter = np.zeros(len(centroid))

        dist = pdist(centroid, all_features)
        #min_dist = dist.min(axis=0)
        min_dist_index = dist.argmin(axis=0)

        for x in range(len(min_dist_index)):
            id = min_dist_index[x]
            new_centroid_coor[id] = np.add(new_centroid_coor[id],
                                          all_features[x])
            new_centroid_counter[id] += 1

        new_centroid_coor_out = []
        for i in range(len(new_centroid_coor)):
            if new_centroid_counter[i] == 0:
                new_centroid_coor_out.append(centroid[i])
            else:
                new_centroid_coor_out.append(np.divide(new_centroid_coor[i],new_centroid_counter[i]))

        return np.array(new_centroid_coor_out), new_centroid_counter


    def _check_convergence_condition(old_centroids, new_centroids, epsilon):
        """
        Check convergence confition

        :param old_centroids:
        :param new_centroids:
        :param epsilon: if every centroid is moved by dist < epsilon KMeans terminates
        :return:
        """
        for i in range(len(old_centroids)):
            dist = euclidean(old_centroids[i], new_centroids[i])
            if dist > epsilon:
                return False

        return True

    def delete_small_clusters(new_centroids, centroid_counter,  threshold):
        """
        Potential extension of the algorithm -> if there is not any point in the cluster, delete this cluste

        :param new_centroids:
        :param centroid_counter:
        :param threshold:
        :return:
        """

        out_centroids = []
        for n in range(len(new_centroids)):
            if centroid_counter[n] > threshold:
                out_centroids.append(new_centroids[n])
        out_centroids = np.array(out_centroids)
        return out_centroids

    #MAIN
    old_centroids = _initiate_random_centroids(all_features, vocab_size)

    for _ in range(max_iter):
        new_centroids, centroid_counter = _assign_vectors_to_nearest_centroid(all_features, old_centroids)
        if_convergenced = _check_convergence_condition(new_centroids, old_centroids, epsilon)

        if if_convergenced == True:
            # return centroids if algorithm is converged
            # return delete_small_clusters(new_centroids, centroid_counter, 0)
            return new_centroids
        old_centroids = new_centroids

    # return centroids if reached max_iter
    # return delete_small_clusters(new_centroids, centroid_counter, 0)
    return new_centroids