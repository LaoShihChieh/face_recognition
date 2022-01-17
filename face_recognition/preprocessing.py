import matplotlib.pyplot as plt
import numpy as np

# load datasets with the first five images from each folder as the training set, and the second five images as the testing set.
def load_datasets():
    file = "att_faces/s"
    
    train_dataset = []
    train_target = []
    test_dataset = []
    test_target = []

    for i in range(1,41):
        for j in range(1,11):
            if j <= 5:
                vector = plt.imread(file + str(i) + "/" + str(j) + ".pgm")
                train_target.append([i])
                train_dataset.append(vector)
            else:
                vector = plt.imread(file + str(i) + "/" + str(j) + ".pgm")
                test_target.append([i])
                test_dataset.append(vector)

    return train_dataset, train_target, test_dataset, test_target

# transform the image in the datasets into 1D and tranform the image and target to arrays
def transform_to_1D_array(train_dataset, train_target, test_dataset, test_target):     
    train_img = []
    test_img = [] 

    for image in train_dataset:
        img_to_1D = []
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                img_to_1D.append(image[x][y])
        train_img.append(img_to_1D)

    for image in test_dataset:
        img_to_1D = []
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                img_to_1D.append(image[x][y])
        test_img.append(img_to_1D)

    train_img = np.array(train_img)
    test_img = np.array(test_img)
    train_target = np.array(train_target)
    test_target = np.array(test_target)

    # use ravel to avoid errors 
    train_target = train_target.ravel()
    test_target = test_target.ravel()

    return train_img, test_img, train_target, test_target
