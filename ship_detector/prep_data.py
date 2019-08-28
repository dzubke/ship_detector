# non-standard libaries
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# standard libraries
from typing import Tuple, List # in standard python library
import glob # in standard python library
import os # in standard python library
import re # in standard python library

def read_images(dir_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Readss the images from directory path dir_path and exports as a np.ndarray with the images unrolled. Each color image has dimensions:
        80 px by 80 px by 3 channels, which means there are a total of 19,200 pixels in each color image. These will be unrolled in the output data_array.
        The label_array only has one column which is a binary of whether the image contains a ship (1) or does not contain a ship (0)

    Parameters
    ----------
    dir_path: str
        The path to the directory that contains the ship images. The directory contains 4,000 images whose filenames begin with '1' if the image contains a
        boat or a '0' if the image does not containg a ship.

    Returns
    ---------   
    data_array: np.ndarray of shape [# images, 19200]
        A numpy array that contains the unrolled images. 
    
    label_array: np.ndarray of shape [# images, 1]
        A numpy array that contains the labels of each image.
    """

    data_list = [] # a list that will contain a list of the image pixels to be converted to a np.ndarray
    label_list = [] # the list that will contain the 1/0 label values

    #the call below makes a lies of all files with the extension
    files_list = glob.glob(dir_path+'*.png')
    positive = re.compile(dir_path+"1.*")
    negative = re.compile(dir_path+"0.*")
    for i in files_list:
        
        #assigns a 1/0 value in the label_array based on the filename
        if positive.match(i):
            label_list.append(1)
        if negative.match(i):
            label_list.append(0)
        
        #opens the file as PIL Image object
        img_temp = Image.open(i)
        data_array_temp_org = np.array(img_temp)
        # the reshape call unravels the 3x80x80 images into a 19200 element column vector
        data_array_temp = data_array_temp_org.reshape(-1)
        data_list.append(data_array_temp)
    
    #in the call below ndmin arg ensures the shape has (X,1) and the '.T' call transposes the array into a column vector
    label_array = np.array(label_list, ndmin=2).T   
    data_array = np.array(data_list)

    return data_array, label_array


def dataset_split(data_array: np.ndarray, label_array: np.ndarray, split_ratios:List[float]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function splits the datasets into training, development, and test sets based on the float percentages given
        in split_ratios

    Parameters
    ----------
    data_array: np.ndarray of shape (# of images x 19200 pixels)
        The input array of flattened images

    label_array: np.ndarray of shape (# of images x 1 )
        The label array of whether a image contains a ship (1) or not (0)

    split_ratios: List of floats with length 3
        Three float values that determine the percentages of how the training, development, and test sets are determined.
        The first value sets the percentage of the dataset in the training set, second flot for the dev set, and so on.
    
    Returns
    --------
    X_train: np.ndarray of shape (# of images * split_ratios[0] x 19200 )
        The training dataset 

    X_dev: np.ndarray of shape (# of images * split_ratios[1] x 19200 )
        The development dataset 

    X_test: np.ndarray of shape (# of images * split_ratios[2] x 19200 )
        The test dataset 
    
     Y_train: np.ndarray of shape (# of images * split_ratios[0] x 1 )
        The training labels
    
    Y_dev: np.ndarray of shape (# of images * split_ratios[1] x 1 )
        The labels for the development set
    
    Y_test: np.ndarray of shape (# of images * split_ratios[2] x 1 )
        The labels of the test set

    """

    assert sum(x for x in split_ratios) <= 1 and sum(x for x in split_ratios) > 0.99, "The sum of the split ratios are too big or small"


    # sklearn.train_test_split only splits the data into two groups. This will be called twice but to ensure the dataset
    # ratios are correct we need to perform the computation below to determine the split ratios of both calls of the splitting function

    split_ratio_1 = 1.0 - split_ratios[1]
    split_ratio_2 = split_ratios[0]/ split_ratio_1 

    X_train, X_dev, Y_train, Y_dev = train_test_split(data_array, label_array, train_size = split_ratio_1, random_state = 1234, stratify = label_array, shuffle = True)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size = split_ratio_2, random_state = 1234, stratify = Y_train, shuffle = True)

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

if __name__ == "__main__":
    pass