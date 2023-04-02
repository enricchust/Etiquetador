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
        if not X.dtype == float: #check if values of matrix X are float
            X = X.astype(float) #if not, it convert them to float

        if len(X.shape) == 3:
            ncols, nrows, _ = X.shape #we unpacks its shape into three variables
            X = X.reshape([ncols * nrows, 3]) #we assign the new shape to the matrix
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

        # Create an empty array of size (self.K, self.X.shape[1]) to store the old centroids
        self.old_centroids = np.zeros((self.K, self.X.shape[1]))

        # Check the initialization method specified in the options dictionary
        if self.options['km_init'].lower() == 'first':
            # Find the first self.K unique rows in self.X
            # Return the indices of the first occurrence of each unique row
            _, idx = np.unique(self.X, axis=0,
                               return_index=True)

            # Sort the indices to obtain the rows in their original order
            idx = np.sort(idx)
            # Extract the unique rows from self.X using the sorted indices
            unique = self.X[idx]  # agafem les files que ens interessen

            # Assign the unique rows as the initial centroids
            self.centroids = unique[:self.K]

        elif self.options['km_init'].lower() == 'random':
            # Find the unique rows in self.X
            # Return the indices of the first occurrence of each unique row
            _, idx = np.unique(self.X, axis=0, return_index=True)

            # Sort the indices to obtain the rows in their original order
            idx = np.sort(idx) #PODEM TREURE???

            # Extract the unique rows from self.X using the sorted indices
            unique = self.X[idx] #PODEM TREURE???

            # Randomly select self.K rows from self.X without replacement
            random_idx = np.random.choice(self.X.shape[0], size=self.K,
                                          replace=False)
            # Assign the randomly selected rows as the initial centroids
            self.centroids = self.X[random_idx]

        elif self.options[
            'km_init'].lower() == 'custom':  # TODO amb la diagonal de l'hipercub
            pass

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """

        arr = distance(self.X, self.centroids)  # N x K
        # t0 = time.perf_counter()
        # getLabels = [np.argmin(row) for row in arr] # argmin() retorna l'índex del valor més petit

        # Find the index of the closest centroid for each point in X
        # getLabels has dimensions (n_samples,)
        getLabels = np.argmin(arr, axis=1)  # argmin() returns the index of the smallest value

        # Assign the computed labels to self.labels
        self.labels = getLabels

        # print(f'\n--> GET_LABELS TIME: {time.perf_counter()-t0}')

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """

        # Save the current centroids as the old centroids
        self.old_centroids = np.copy(self.centroids)

        # Create a dictionary with keys as cluster labels (0 to K-1)
        # and empty lists as values to store the coordinates of each point in each cluster
        centroidsDict = {i: [] for i in range(self.K)}

        # Assign each point to its corresponding cluster
        for label, row in zip(self.labels, self.X):
            centroidsDict[label].append(row)

        # Compute the mean coordinates of all points in each cluster
        centroids = [np.mean(value, axis=0) for value in centroidsDict.values()]

        # Convert the list of mean coordinates into a NumPy array with dimensions (K, C)
        # where K is the number of clusters and C is the number of features in the dataset
        self.centroids = np.array(centroids).reshape(self.K, self.X.shape[1])

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """

        # Check if the centroids have converged based on the tolerance specified in the options dictionary
        # The centroids are considered to have converged if the absolute difference between the old centroids
        # and the new centroids is less than or equal to the tolerance
        # Return True if the centroids have converged, False otherwise
        # The iterations are checked in the fit() function
        # Using np.allclose() instead of calculating the absolute difference and checking if all elements are less than or equal to the tolerance
        # can be more numerically stable when dealing with small values or arrays with different precisions
        return np.allclose(self.old_centroids, self.centroids)


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """

        # Initialize the centroids using the method specified in the options dictionary
        self._init_centroids()

        # Run the K-Means algorithm until convergence or until the maximum number of iterations is reached
        while not self.converges() and self.num_iter < self.options['max_iter']:
            # Assign each point in self.X to the closest centroid
            self.get_labels()
            # Compute the mean coordinates of all points in each cluster to obtain the new centroids
            self.get_centroids()
            # Increment the number of iterations performed so far
            self.num_iter += 1

        print(f'\nNUMBER OF FIT ITERATIONS: {self.num_iter}')

    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering

        """
        self.WCD = sum(np.sqrt(
            np.sum((self.X - self.centroids[self.labels]) ** 2, axis=1))) / \
                   self.X.shape[0]

    def find_bestK(self, max_K):
        """
        sets the best k anlysing the results up to 'max_K' clusters
        """

        self.K = 1
        self.fit()
        WCD =  self.withinClassDistance()

        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            self.withinClassDistance()
            actWCD = self.withinClassDistance()

            if 100 - 100 * actWCD / WCD < self.options['DEC_threshold']:
                self.K = k - 1
                break
            WCD = actWCD

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

    dist = np.empty((X.shape[0], C.shape[0]))

    for idx, x in enumerate(X):
        #dist[:, idx] = np.power(np.sum((X - centroids) ** 2, axis=1), 1 / 2)  # dist[:,idx] assigna la distancia de cada centroide a una columna
        #SI EL DE BAIX NO XUTA, CAMBIAR EL X del for per C
        dist[idx] = np.linialg.norm(x - C, ord=2, axis=1)

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


def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
