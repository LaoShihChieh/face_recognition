# Base on SVM on Face Recognize with PCA&LDA Dimension Reduction
### ***There are many theories of PCA, LDA, SVM on Internet, so I will not explain the background knowledge in here, instead , I will focus on my codes and methods.***
## I. Data
***Every face has 92 * 112 = 10304 pixels with grayscale values and is saved to pgm file, 10 pieces/per classes, 40 classes, 400 pgm files in total.***

![](https://i.imgur.com/YFj3DB8.png)

## II. Data Preprocessing
***I used PIL library for getting grayscale values and restored to an one-dimension (1, 10304) martix.***
``` python
from PIL import Image


# Convert pgm file to vector of grayscale value
def read_img(pgm):
    img = Image.open(pgm)    # read the pgm file
    pixels = img.load()
    vector = []
    for j in range(0, img.size[1]):
        for i in range(0, img.size[0]):
            vector.append(pixels[i, j])  # get row first
    return vector   # output (1, 10304)

```
***Divided data in 200 train data and 200 test data, and then saved in two 2-dimension(200, 10304) matrix.***
```python
# Get train datasets
def load_training_datasets():
    file = "att_faces/s"
    # Get 200 pgm files to create training data (200, 10304).
    training_dataframe = []
    # Get the class label of 200 pgm files (200, 1).
    training_target = []
    for i in range(1, 41):
        for j in range(1, 6):
            vector = read_img(file + str(i) + "/" + str(j) + ".pgm")
            training_target.append([i])
            training_dataframe.append(vector)
    return training_dataframe, training_target


# Get test datasets
def load_test_datasets():
    file = "att_faces/s"
    # Get 200 pgm files to create training data (200, 10304).
    test_dataframe = []
    # Get the class label of 200 pgm files (200, 1).
    test_target = []
    for i in range(1, 41):
        for j in range(6, 11):
            vector = read_img(file + str(i) + "/" + str(j) + ".pgm")
            test_target.append([i])
            test_dataframe.append(vector)
    return test_dataframe, test_target

```
## III. Main Process
***Import the library we need.***
``` python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import Data_Preprocessing
import numpy as np
import pandas as pd
```
***Define a function to do PCA transform directly.***
* ***dimensions - The dimension reduction we woud like.***
* ***training_data - The data from load_training_datasets() in Data Preprocessing without classes.***
* ***test_data - The data from load_test_datasets() in Data Preprocessing without classes.***
 
***Because PCA is unsupervised learning, so we don't need train_target(classes) in model.fit().***
```python
def pca_transform(dimensions, training_data, test_data):
    # Construct PCA model
    model = PCA(n_components=dimensions)

    # Compute mean of every picture
    training_data = np.array(training_data)
    training_data_mean = training_data.mean(axis=0)

    # Normalize to make E[x] = 0
    training_data_zero_mean = training_data - training_data_mean

    # Model Training
    model.fit(training_data_zero_mean)

    # Transform test data
    test_transform = model.transform(test_data - training_data_mean)
    return model, test_transform, training_data_zero_mean

```
***Define a function to do LDA transform directly.***
* ***dimensions - The dimension reduction we woud like.***
* ***training_data - The data from load_training_datasets() in Data Preprocessing without classes.***
* ***training_target - The classes from load_training_datasets() in Data Preprocessing .***
* ***test_data - The data from load_test_datasets() in Data Preprocessing without classes.***
 
***Because LDA is supervised learning, so we do need train_target(classes) in model.fit().***
```python
def lda_transform(dimensions, training_data, training_target, test_data):
    # Construct LDA model
    model = LinearDiscriminantAnalysis(n_components=dimensions)

    # Compute mean of every picture
    training_data = np.array(training_data)
    training_data_mean = training_data.mean(axis=0)

    # Normalize to make E[x] = 0
    training_data_zero_mean = training_data - training_data_mean

    # Model Training
    model.fit(training_data_zero_mean, training_target)

    # Transform test data
    test_transform = model.transform(test_data - training_data_mean)
    return model, test_transform, training_data_zero_mean

```
***In line 20(or all process about LDA), you can notice the if-condition written "if i==40", due to Sklearn LDA model can just accept the dimension between 0 to min(n_features, n_classes-1), and we have 40 classes, so the max dimension is 39.***
```python
if __name__ == "__main__":
    # From Data_Preprocessing load in train data & test data
    training_datasets, training_target = Data_Preprocessing.load_training_datasets()
    test_datasets, test_target = Data_Preprocessing.load_test_datasets()

    # Ravel target data for avoiding errors
    training_target = np.array(training_target)
    training_target = training_target.ravel()
    test_target = np.array(test_target)
    test_target = test_target.ravel()

    # Dimension reduction
    pca_dimension = [10, 20, 30, 40, 50]
    lda_dimension = [10, 20, 30]
    PCA_true_counter_list = []
    PCA_false_counter_list = []
    LDA_true_counter_list = []
    LDA_false_counter_list = []
    for i in pca_dimension:
        if i == 40:
            print("\n\nNow, dimension is larger than class label=40, so LDA is stopped working. \n\n")

        # PCA
        pca, pca_test, pca_training_zero_mean = pca_transform(i, training_datasets, test_datasets)

        # LDA
        if i < 40:
            lda, pca2lda, lda_training_zero_mean = lda_transform(
                i, pca.transform(pca_training_zero_mean), training_target, pca_test)

        # Construct SVM Model
        SVM = SVC(kernel="linear")

        # SVM for PCA
        SVM.fit(pca.transform(pca_training_zero_mean), training_target)
        SVM_PCA_predict_result = SVM.predict(pca_test)

        # SVM for LDA
        if i < 40:
            SVM.fit(lda.transform(lda_training_zero_mean), training_target)
            SVM_LDA_predict_result = SVM.predict(pca2lda)

        # Count PCA Confusion Matrix
        SVM_PCA_confusion_matrix = confusion_matrix(test_target, SVM_PCA_predict_result)

        # Count LDA Confusion Matrix
        if i < 40:
            SVM_LDA_confusion_matrix = confusion_matrix(test_target, SVM_LDA_predict_result)

        # Print accuracy of PCA
        print("Accuracy of dimension {:d} in SVM for PCA : {:.2f}%".format(
            i, accuracy_score(SVM_PCA_predict_result, test_target) * 100))
        print("---------------------------------------")

        # Print accuracy of LDA
        if i < 40:
            print("Accuracy of dimension {:d} in SVM for LDA : {:.2f}%".format(
                i, accuracy_score(SVM_LDA_predict_result, test_target) * 100))
            print("---------------------------------------")

        # Print Confusion Matrix of PCA
        print("Confusion Matrix of dimension {:d} in SVM for PCA : \n".format(i), SVM_PCA_confusion_matrix)
        print("---------------------------------------")

        # Print Confusion Matrix of LDA
        if i < 40:
            print("Confusion Matrix of dimension {:d} in SVM for PCA : \n".format(i), SVM_LDA_confusion_matrix)
            print("=======================================\n")
        else:
            print("=======================================\n")

        # Count PCA true & false data
        true_counter = 0
        false_counter = 0
        for m in range(0, len(test_target)):
            if SVM_PCA_predict_result[m] == test_target[m]:
                true_counter += 1
            else:
                false_counter += 1
        PCA_true_counter_list.append(true_counter)
        PCA_false_counter_list.append(false_counter)

        # Count LDA true & false data
        if i < 40:
            true_counter = 0
            false_counter = 0
            for m in range(0, len(test_target)):
                if SVM_LDA_predict_result[m] == test_target[m]:
                    true_counter += 1
                else:
                    false_counter += 1
            LDA_true_counter_list.append(true_counter)
            LDA_false_counter_list.append(false_counter)

    # Jump out the loop
    # Use dataframe to create PCA Confusion Data
    PCA_statistics_dictionary = {"dimension": pca_dimension,
                                 "True item": PCA_true_counter_list,
                                 "False item": PCA_false_counter_list
                                 }
    PCA_statistics_dataframe = pd.DataFrame(PCA_statistics_dictionary)
    print(PCA_statistics_dataframe)

    print("\n=======================================\n")

    # Use dataframe to create PCA Confusion Data
    LDA_statistics_dictionary = {"dimension": lda_dimension,
                                 "True item": LDA_true_counter_list,
                                 "False item": LDA_false_counter_list
                                 }
    LDA_statistics_dataframe = pd.DataFrame(LDA_statistics_dictionary)
    print(LDA_statistics_dataframe)

```
## IV. Final Result
***PCA bases on the features that possess the top d(dimension we need) largest feature contribution.***

***LDA bases on the features that possess the top d(dimension we need) largest class separation.***

***So, when in low dimension, LDA is better than PCA, on the contrary, PCA is more precise when dimension is higher.***
```
Accuracy of dimension 10 in SVM for PCA : 86.50%
---------------------------------------
Accuracy of dimension 10 in SVM for LDA : 88.50%
---------------------------------------
Confusion Matrix of dimension 10 in SVM for PCA : 
 [[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
---------------------------------------
Confusion Matrix of dimension 10 in SVM for PCA : 
 [[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
=======================================

Accuracy of dimension 20 in SVM for PCA : 88.00%
---------------------------------------
Accuracy of dimension 20 in SVM for LDA : 87.00%
---------------------------------------
Confusion Matrix of dimension 20 in SVM for PCA : 
 [[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
---------------------------------------
Confusion Matrix of dimension 20 in SVM for PCA : 
 [[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 5]]
=======================================

Accuracy of dimension 30 in SVM for PCA : 89.00%
---------------------------------------
Accuracy of dimension 30 in SVM for LDA : 87.50%
---------------------------------------
Confusion Matrix of dimension 30 in SVM for PCA : 
 [[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
---------------------------------------
Confusion Matrix of dimension 30 in SVM for PCA : 
 [[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 5]]
=======================================



Now, dimension is larger than class label=40, so LDA is stopped working. 


Accuracy of dimension 40 in SVM for PCA : 89.50%
---------------------------------------
Confusion Matrix of dimension 40 in SVM for PCA : 
 [[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
---------------------------------------
=======================================

Accuracy of dimension 50 in SVM for PCA : 90.00%
---------------------------------------
Confusion Matrix of dimension 50 in SVM for PCA : 
 [[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
---------------------------------------
=======================================

Confusion Data of PCA and LDA : 

   dimension  True item  False item
0         10        173          27
1         20        176          24
2         30        178          22
3         40        179          21
4         50        180          20

---------------------------------------

   dimension  True item  False item
0         10        177          23
1         20        174          26
2         30        175          25
```
