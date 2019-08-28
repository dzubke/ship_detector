# all of the local modules
from prep_data import read_images, dataset_split
from explore_data import array_info, image_info

def main():
    """The place where I put all the 'glue-code' that calls all the various functions together
    """

    dir_path =r'/Users/dustin/CS/projects/ship_detector/data/ships-in-satellite-imagery/shipsnet/'

    data_array, label_array = read_images(dir_path)

    array_info(data_array, label_array)

    image_info(data_array[0,:], plot_image=False)

    split_ratios = [0.8, 0.1, 0.1]      #splitting the dataset into 80% train, 10% dev, 10% test

    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = dataset_split(data_array, label_array, split_ratios)





if __name__ == "__main__":
    main()