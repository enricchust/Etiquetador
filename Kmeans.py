__authors__ = '1637620, 1638322, 1638529'
__group__ = 'Grup09'

import numpy as np
import utils
import time


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        if not X.dtype == float:
            X = X.astype(float)

        if len(X.shape) == 3:
            ncols, nrows, _ = X.shape
            X = X.reshape([ncols * nrows, 3])
            self.X = X
            return X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        self.options = options

    def _init_centroids(self):

        self.old_centroids = np.zeros((self.K, self.X.shape[1]))
        if self.options['km_init'].lower() == 'first':
            _, idx = np.unique(self.X, axis=0,
                               return_index=True)  # Retorna els index corresponents a files úniques
            idx = np.sort(
                idx)  # els ordenem per obtenir les files en l'ordre original
            unique = self.X[idx]  # agafem les files que ens interessen
            self.centroids = unique[:self.K]

        elif self.options['km_init'].lower() == 'random':
            _, idx = np.unique(self.X, axis=0, return_index=True)
            idx = np.sort(idx)
            unique = self.X[idx]
            random_idx = np.random.choice(self.X.shape[0], size=self.K,
                                          replace=False)  # agafem K files aleatòries sense repetició
            self.centroids = self.X[random_idx]

        elif self.options[
            'km_init'].lower() == 'custom':  # TODO amb la diagonal de l'hipercub
            pass

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """

        arr = distance(self.X, self.centroids)  # N x K
        getLabels = np.argmin(arr, axis=1)
        self.labels = getLabels

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """

        self.old_centroids = np.copy(self.centroids)
        centroidsDict = {i: [] for i in range(
            self.K)}  # Diccionari amb els clusters com a claus = (0,...,K-1)

        for label, row in zip(self.labels,
                              self.X):  # Posem cada punt al cluster corresponent
            centroidsDict[label].append(row)

        centroids = [np.mean(value, axis=0) for value in
                     centroidsDict.values()]  # Fem la mitjana dels punts de cada clúster
        self.centroids = np.array(centroids).reshape(self.K, self.X.shape[
            1])  # Ho posem en la forma adequada (K x C)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """

        # .all() comprova que tots els elements compleixin la condició
        # return abs(self.old_centroids - self.centroids).all() <= self.options['tolerance'] # les iteracions es miren a la funció fit
        return np.allclose(self.old_centroids, self.centroids)

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()
        while not self.converges() and self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1

    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """
        wcd = 0
        for i, centroid in enumerate(self.centroids):
            idxs = np.where(self.labels == i)[
                0]  # Choose the indexes of the i-th cluster
            idx_matrix = self.X[idxs]
            dist_array = distance(idx_matrix, np.array(
                [centroid]))  # Calculate the distance from the centroid
            wcd += np.sum(dist_array ** 2)
        self.WCD = wcd / self.X.shape[
            0]  # Compute the averge of each WCD from each cluster
        return self.WCD

    def find_bestK(self, max_K):
        """
        sets the best k anlysing the results up to 'max_K' clusters
        """
        if self.options['fitting'] == 'WCD':
            self.K = 1
            self.fit()
            self.withinClassDistance()
            WCDk_1 = self.WCD

            for k in range(2, max_K + 1):
                self.K = k
                self.fit()
                self.withinClassDistance()
                WCDk = self.WCD

                DECk = WCDk / WCDk_1
                WCDk_1 = WCDk

                if 1 - DECk <= 0.2:
                    self.K = k - 1
                    return k - 1

            self.K = max_K

        if self.options['fitting'] == 'Silhouette':
            pass

        if self.options['fitting'] == 'Fisher Ratio':
            pass


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    dist = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    colorProb = utils.get_color_prob(
        centroids)  # retorna matriu K x 11 on cada columna és P(color)
    idx = np.argmax(colorProb,
                    axis=1)  # array de K valors on hi ha els índexs amb probabilitat més alta
    return utils.colors[idx]  # Agafem els colors corresponents als índexs