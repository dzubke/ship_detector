# standard libraries
import time

# non-standard libraries
from sklearn.linear_model import LogisticRegression


# all of the local modules
from prep_data import read_images, dataset_split
from explore_data import array_info, image_info
from models import run_model
from assess_model import count_time, roc_assess, F1score_assess


def main():
    """The place where I put all the 'glue-code' that calls all the various functions together
    """

    dir_path =r'/Users/dustin/CS/projects/ship_detector/data/ships-in-satellite-imagery/shipsnet/'

    data_array, label_array = read_images(dir_path)

    array_info(data_array, label_array)

    image_info(data_array[0,:], plot_image=False)

    split_ratios = [0.8, 0.1, 0.1]      #splitting the dataset into 80% train, 10% dev, 10% test

    Xtrain, Xdev, Xtest, ytrain, ydev, ytest = dataset_split(data_array, label_array, split_ratios)

    print(f"xtrain, xdev, xtest, ytrain, ydev, ytest shapes: {Xtrain.shape}, {Xdev.shape}, {Xtest.shape}, {ytrain.shape}, {ydev.shape} {ytest.shape} ")

    print(type(LogisticRegression()))

    model = LogisticRegression(solver='lbfgs')

    model_fit = model.fit(Xtrain, ytrain)

    train_acc=model_fit.score(Xtrain, ytrain)
    test_acc=model_fit.score(Xtest,ytest)
    print("Training Data Accuracy: %0.2f" %(train_acc))
    print("Test Data Accuracy:     %0.2f" %(test_acc))

    roc_assess(model_fit, Xtest, ytest, print_values=True)
    F1score_assess(model_fit, Xtest, ytest, print_values=True)



    # run_model(logreg, Xtrain, Xdev, ytrain, ydev)

    # output = count_time(run_model(Xtrain, Xdev, ytrain, ydev))


    # print(output[0])


if __name__ == "__main__": main()