__authors__ = '1637620, 1638322, 1638529'
__group__ = 'Grup09'

import numpy as np
import utils

# TODO: On posem l'array
colors = np.array(
    ['Red', 'Orange', 'Brown', 'Yellow', 'Green', 'Blue', 'Purple', 'Pink',
     'Black', 'Grey', 'White'])


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

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

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
            # TODO: mirar què fer en el else

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

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'first':
            _, idx = np.unique(self.X, axis=0,
                               return_index=True)  # Retorna els index corresponents a files úniques
            idx = np.sort(
                idx)  # els ordenem per obtenir les files en l'ordre original
            unique = self.X[idx]  # agafem les files que ens interessen
            self.centroids = unique[:self.K]

        elif self.options['km_init'].lower() == 'random':  # TODO
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
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        arr = distance(self.X, self.centroids)  # N x K
        getLabels = [np.argmin(row) for row in
                     arr]  # argmin() retorna l'índex del valor més petit

        self.labels = getLabels

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = self.centroids
        centroidsDict = {i: [] for i in range(
            self.K)}  # Diccionari els clusters com a claus = (0,...,K-1)

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
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        # .all() comprova que tots els elements compleixin la condició
        return abs(self.old_centroids - self.centroids).all() <= self.options[
            'tolerance']  # les iteracions es miren a la funció fit

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.random.rand()

    def find_bestK(self, max_K):
        """
        sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
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

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    arr = np.array([euclidean_dist(x, c) for x in X for c in C])
    return np.reshape(arr, [X.shape[0], C.shape[0]])


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    colorProb = utils.get_color_prob(
        centroids)  # retorna matriu K x 11 on cada columna és P(color)
    idx = np.argmax(colorProb,
                    axis=1)  # array de K valors on hi ha els índexs amb probabilitat més alta
    return colors[idx]  # Agafem els colors corresponents als índexs


def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))