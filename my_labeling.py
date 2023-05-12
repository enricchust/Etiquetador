__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import matplotlib.pyplot as plt

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

    # load our knn

    knn = KNN(train_imgs, train_class_labels)


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
            Return a list of the index that have these colors
        """
        indicesToPrint = []
        indicesAcurate = []

        for index, element in enumerate(predicted_colors):
            auxList = np.isin(search, element)
            if np.all(auxList): # check that all the colors we want are in the sample
                #print(index)
                indicesToPrint.append(index)
                indicesAcurate.append(element)
        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)
        return indicesToPrint, indicesAcurate

    #print(retrieval_by_color(train_imgs, train_color_labels, ["Blue", "White"], 20))
    #knn = KNN(train_imgs, train_labels)

    def retrieval_by_shape(list_images, predicted_shapes, search, n):
        """
        Args:
            list_images: dataset of the images, obtained by the ground truth
            predicted_shapes: list of the colors we have obtained after aply the KNN
            search: colors we want to search in the images

        Return:
            Return a list of the indexs that have these shapes
        """

        indicesToPrint = []
        indicesAcurate = []

        for index, element in enumerate(predicted_shapes):
            if search in element: # check that all the colors we want are in the sample
                print(index)
                indicesToPrint.append(index)
                indicesAcurate.append(element)
        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)
        return imagesGiven


    knn = KNN(train_imgs, train_class_labels)
    label_results = knn.predict(test_imgs, 10)
    #pred = knn.predict(test_imgs, test_class_labels)
    #preds = knn.predict(test_imgs['test_input'][0], test_imgs['rnd_K'])
    #print(pred)
    im = retrieval_by_color(test_imgs, label_results, "Shorts", 20)
    print(im)
    #plt.imshow(im[0])
    #load_imgs(train_imgs, test_imgs)
    #read_one_img(im[0])


    def retrieval_combined(list_images, predicted_colors, predicted_shapes, search_color, search_shape, n):
        indicesToPrint = []

        for index, color in enumerate(predicted_colors):
            auxList = np.isin(search_color, color)
            if np.all(auxList):  # check that all the colors we want are in the sample
                if search_shape in predicted_shapes[index]:
                    indicesToPrint.append(index)
        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)
        return indicesToPrint

    #print(retrieval_combined(train_imgs, test_color_labels, test_class_labels, ["Blue"], "Shorts", 5))









