__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import matplotlib.pyplot as plt

from utils_data import *
import numpy as np
from KNN import *
from Kmeans import *
from TestCases_kmeans import *
import random

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
    test_color_labels = read_dataset(root_folder='./images/',
                                     gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here


    # load ou kmeans
    #km = KMeans(train_imgs)
    #color_results = km.fit()
    """
    for ix, input in enumerate(self.test_cases['input']):
        km = KMeans(input, self.test_cases['K'][ix])
    """
    # load our knn
    knn = KNN(train_imgs, train_class_labels)
    label_results = knn.predict(test_imgs, 6)


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

        if type(search) != list: search = [search]

        for index, element in enumerate(predicted_colors):
            auxList = np.isin(search, element)
            if np.all(auxList): # check that all the colors we want are in the sample
                indicesToPrint.append(index)

        random.shuffle(indicesToPrint)
        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)
        return indicesToPrint

    colors = []
    for im in test_imgs[:300]:
        km = KMeans(im, 4)
        km.fit()
        colors.append(get_colors(km.centroids))
        #print(get_colors(km.centroids))

    #print(retrieval_by_color(test_imgs, colors, ["Black"], 10))
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

        for index, element in enumerate(predicted_shapes):
            if search in element: # check that all the colors we want are in the sample
                print(index)
                indicesToPrint.append(index)

        random.shuffle(indicesToPrint)
        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)
        return imagesGiven

    #im = retrieval_by_shape(test_imgs, label_results, "Handbags", 10)
    #im2 = retrieval_by_shape(test_imgs, label_results, "Dresses", 10)
    #print(im)



    #pred = knn.predict(test_imgs, test_class_labels)
    #preds = knn.predict(test_imgs['test_input'][0], test_imgs['rnd_K'])
    #print(pred)
    #plt.imshow(im[0])
    #load_imgs(train_imgs, test_imgs)
    #read_one_img(im[0])


    def retrieval_combined(list_images, predicted_colors, predicted_shapes, search_color, search_shape, n):
        indicesColor = []
        indicesToPrint = []

        for index, color in enumerate(predicted_colors):
            auxList = np.isin(search_color, color)
            if np.all(auxList):  # check that all the colors we want are in the sample
                indicesColor.append(index)

        list_images = list_images[indicesColor]

        predicted_shapes = predicted_shapes[indicesColor]

        for index, shape in enumerate(predicted_shapes):
            if shape in search_shape:
                indicesToPrint.append(index)

        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)

        return indicesToPrint


    #print(retrieval_combined(train_imgs, colors, label_results, ["Blue"], "Dresses"))
    #print(retrieval_combined(train_imgs, colors, label_results, ["Blue"], "Dresses", 5))

    def retrieval_by_colors_combination(list_images, predicted_colors, search, n, search_type=1):
        """
        Args:
            list_images: dataset of the images, obtained by the ground truth
            predicted_colors: list of the colors we have obtained after aply the K-means
            search: combination of colors we want to search in the images
            search_type: you can choose if you want samples where there is only all this colors
                        or samples where there are combinations of this colors but there is no need
                        of all the colors in the same sample.
                        If search_type = 0, it shows only samples that have all the colors.
                        If search_type = 1, it shows all the samples that have at least two of this
                        colors, but ordered by more colors.
                        If search_type = 2, it shows all the samples that have at least two of this
                        colors randomly.


        Return:
            Return a list of the index that have these colors
        """
        indicesToPrint = []
        indicesAux = {}


        if type(search) != list: search = [search]

        for index, element in enumerate(predicted_colors):
            auxList = np.isin(search, element)
            #if(np.sum(auxList)>1):
                #print(np.sum(auxList))
            if np.all(auxList): # check that all the colors we want are in the sample
                indicesToPrint.append(index)

            elif np.sum(auxList) >= 2 and search_type != 0:
                if(search_type == 1):
                    key = np.sum(auxList)
                    #print(1)
                    if key in indicesAux:
                        print(2)
                        indicesAux[key].append(index)
                    else:
                        print(3)
                        indicesAux[key] = [index]

                else:
                    indicesToPrint.append(index)

        if search_type == 1:
            # Sort the keys in descending order and reverse the order
            sorted_keys = sorted(indicesAux.keys(), reverse=True)
            for key in reversed(sorted_keys):
                list_to_merge = indicesAux[key]
                for i in list_to_merge:
                    indicesToPrint.append(i)

        #random.shuffle(indicesToPrint)
        imagesGiven = list_images[indicesToPrint]
        visualize_retrieval(imagesGiven, n)

        return indicesToPrint

    retrieval_by_colors_combination(test_imgs, test_color_labels, ["Blue", "Yellow", "Orange", "White"], 10, 2)









