# This file contains functions that will be used to understand the dataset analyzed

# non-standard libraries
import matplotlib.pyplot as plt     
import numpy as np


def array_info(data_array: np.ndarray, label_array: np.ndarray) -> None: 
    """This fuction provides information on the labels of the array.

    Parameters
    ----------
    data_array: np.ndarray of shape (# of images by 19200)
        Contains all of the flattened images from the dataset
    
    label_array: np.ndarray of shape (# of images by 1)
        Contains all of the label information of dataset

    Returns
    --------
    None
        This function doesn't return anything.

    """

    print("\n==========ARRAY_INFO===============")
    print("--------Information on the image data array----------")
    print(f"The data array has shape: {data_array.shape}")
    print(f"There are {data_array.shape[0]} images in the original dataset")
    print(f"Each image is colored (3 channels, RedGreenBlue) and is 80 px by 80 px")
    print(f"When flattened the total number of pixels in each image is {data_array.shape[1]}")

    print("\n-------Information on the label array---------")
    print(f"There are {len(label_array)} image labels in the array.")
    print(f"{sum(x==1 for x in label_array)} of those images have ships in them.")
    print(f"{sum(x==0 for x in label_array)} of those images do NOT have ships in them.")



def image_info(img_array: np.ndarray, plot_image: bool = True) -> None:
    """This function takes in a flatten image numpy array as input, reshapes the array into a 3 channel 80 px by 80px image
     and prints off various kinds of information about the specific image. If plot_image = True, then it will also plot the image.

    Parameters
    ----------
    img_array: np.ndarray of shape (1, 19200)
        a row vector of the flattened 3 channel 80 px by 80 px image

    Returns
    --------
    None
        This function doesn't return anything
    """

    assert img_array.shape == (19200,), "The flattened image array is not the expected shape"

    img_reshape = img_array.reshape(80,80,3)

    print("\n==========IMAGE_INFO===============")
    print(f"The color image has {img_reshape.shape[2]} channels (colors)")
    print(f"The image is {img_reshape.shape[0]} pixels long and {img_reshape.shape[1]} pixels wide")

    if plot_image==True:
        print("\nPlotting the image")
        plt.imshow(img_reshape)
        plt.show()
    elif plot_image==False:
        print("\nNot plotting the image")
    else:
        raise ValueError("plot_image needs to be bool type")