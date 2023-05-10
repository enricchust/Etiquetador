__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import *
import numpy as np
from KNN import *

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # You can start coding your functions here

    """"
    def get_colors(objective_colors, prediction):
        #we have thought that you can search by two colors, so we put the colors into a list if it's alone
        if type(objective_colors) != list:
            objective_colors = [objective_colors]

        n_objective_colors = len(objective_colors)
        coincidence_values = []

        for i in prediction:
            tempList = np.isin(objective_colors, i, assume_unique=False,invert=False)
            percentage = np.sum[tempList] / n_objective_colors
            if percentage != 0:
                coincidence_values.append(i)
    """

    def retrieval_by_color(list_images, predicted_colors, search, n):
        """
        Args:
            list_images: dataset of the images, obtained by the ground truth
            predicted_colors: list of the colors we have obtained after aply the K-means
            search: colors we want to search in the images

        Return:
            Return a list of the indexs that have these colors
        """
        imagesToPrint = []

        for index, element in enumerate(predicted_colors):
            auxList = np.isin(search, element)
            if np.all(auxList): # check that all the colors we want are in the sample
                #print(index)
                imagesToPrint.append(index)
        imagesGiven = test_imgs[imagesToPrint]
        return visualize_retrieval(imagesGiven, n)

    retrieval_by_color(train_color_labels, test_color_labels, ["Blue"], 20)
    #knn = KNN(train_imgs, train_labels)

    def retrieval_by_shape(list_images, predicted_shapes, search, n):
        """
        Args:
            list_images: dataset of the images, obtained by the ground truth
            predicted_colors: list of the colors we have obtained after aply the K-means
            search: colors we want to search in the images

        Return:
            Return a list of the indexs that have these shapes
        """

        imagesToPrint = []

        if type(search) != list: search = [search]

        for index, element in enumerate(predicted_shapes):
            auxList = np.isin(search, element)
            if np.sum(auxList) == len(
                    search):  # all the colors are in that sample
                # indexes.append(index)
                imagesToPrint.append(list_images[index])
        return visualize_retrieval(imagesToPrint, n)









