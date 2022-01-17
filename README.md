![](https://i.imgur.com/hivrzTC.jpg)
# Pattern Recognition Assignment 2
###### `by 勞士杰 資工碩一 611021201       (discussed with 吳承翰)`
---
## Problem: Face Recognition
---
HackMD: https://hackmd.io/0SE-DxmMTZi1RxBmZisFew
## Contents
**1. Problem description**

**2. Preprocess**

**3. Final Code**

**4. Discussion**

**5. Summary**

---
## 1. Problem description
In this assignment, we are using the following face image database to design a face recognition program using PCA and LDA.
![](https://i.imgur.com/OjFde1l.png)

The database that is found [here](https://mitlab.ndhu.edu.tw/~ccchiang/PR/att_faces.zip) contains 400 images (40 subjects, 10 images per subject).

## 2. Preprocess
Every face has 92 x 112 = 10304 pixels with grayscale values and is saved to pgm file. We first use the PIL library to get the grayscale value from the pgm file. Then I used the first five images from each folder as the training set, and the second five images as the testing set.

```python=
def load_datasets():
    file = "att_faces/s"
    
    train_dataset = []
    train_target = []
    test_dataset = []
    test_target = []
    
    for i in range(1,41):
        for j in range(1,11):
            if j <= 5:
                vector = read_img(file + str(i) + "/" + str(j) + ".pgm")
                train_target.append([i])
                train_dataset.append(vector)
            else:
                vector = read_img(file + str(i) + "/" + str(j) + ".pgm")
                test_target.append([i])
                test_dataset.append(vector)
    
    return train_dataset, train_target, test_dataset, test_target
```
Next we transform the image in the datasets into 1D and tranform the image and target to arrays.

```python=
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
```

## 3. Final code
First we import the library that we need to use.
```python=
import preprocessing
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
```
Next we load the datasets and transform them into 1D array.  We then set the parameters for SVC.
```python=
# load datasets and transform them into 1D array
train_dataset, train_target, test_dataset, test_target = preprocessing.load_datasets()
train_img, test_img, train_target, test_target = preprocessing.transform_to_1D_array(train_dataset, train_target, test_dataset, test_target)

# set the parameters for SVC
pca_clf = SVC(kernel='linear', class_weight='balanced')
lda_clf = SVC(kernel='linear', class_weight='balanced')
```
We apply the PCA model with 10,20,30,40,50 dimensions. For each dimension, we do the PCA fit and PCA transform. 

Because PCA is **unsupervised learning**, so we **don’t need** the train_target in PCA.fit().

Then we make the prediction and get the accuracy.

```python=
for i in range(10,51,10):
    print()
    print('{}dimensions:'.format(i))

    # because PCA is unsupervised learning, so we don’t need train_target in PCA.fit().
    PCA_model = PCA(n_components=i).fit(train_img)

    # PCA transform
    pca_train_img = PCA_model.transform(train_img)
    pca_test_img = PCA_model.transform(test_img)

    # prediction
    pca_pred = pca_clf.fit(pca_train_img, train_target)
    pca_test_pred = pca_pred.predict(pca_test_img)
    pca_result = confusion_matrix(test_target, pca_test_pred)

    # accuracy
    pca_accuracy = str(accuracy_score(pca_test_pred,test_target)*100)+"%"
    pca_report = classification_report(test_target,pca_test_pred)

    # count the correct predictions
    true_count = 0
    for j in range(len(pca_result)):
        true_count += pca_result[j][j]
        
    # count the wrong predictions
    false_count = 200 - true_count
```

We apply the LDA model from 0 to 39 dimensions because the model only takes 0~min(n_features, n_classes-1). 

For each dimension, we do the LDA fit and LDA transform.
Because LDA is **supervised learning**, so we **need** the train_target in LDA.fit().

Then we make the prediction and get the accuracy.

```python=
    # apply the LDA model after the PCA with 0 to 39 dimensions
    if(i<40):
        # Because LDA  is supervised learning, so we need the train_target in LDA.fit().
        LDA_model = LDA(n_components=i).fit(pca_train_img, train_target)

        # LDA transform the PCA data
        lda_train_img = LDA_model.transform(pca_train_img)
        lda_test_img = LDA_model.transform(pca_test_img)
        
        # prediction
        lda_pred = lda_clf.fit(lda_train_img, train_target)
        lda_test_pred = lda_pred.predict(lda_test_img)
        lda_result = confusion_matrix(test_target, lda_test_pred)

        # accuracy
        lda_accuracy = str(accuracy_score(lda_test_pred, test_target) * 100) + "%"
        lda_report = classification_report(test_target,lda_test_pred)
        # count the correct predictions
        true_count = 0
        for j in range(len(lda_result)):
            true_count += lda_result[j][j]

        # count the wrong predictions
        false_count = 200 - true_count
```

Finally we print the results.
```python=
# show the performance evaluation
        print('PCA Correct Predictions:{} Wrong Predictions:{} Accuracy:{}'.format(true_count,false_count,pca_accuracy))
        print('LDA Correct Predictions:{}  Wrong Predictions:{} Accuracy:{}'.format(true_count, false_count, lda_accuracy))
        
        # show the confusion matrix
        print('PCA Confusion Matrix:')
        print(pca_result)
        print('LDA Confusion Matrix:')
        print(lda_result)

        # show the report
        print('PCA report:')
        print(pca_report)
        print('LDA report:')
        print(lda_report)
        
    else:
        # show the performance evaluation
        print('PCA Correct Predictions:{} Wrong Predictions:{} Accuracy:{}'.format(true_count,false_count,pca_accuracy))
        
        # show the confusion matrix
        print('PCA Confusion Matrix:')
        print(pca_result)

        # show the report
        print('PCA report:')
        print(pca_report)

```
### Results
We can find that for PCA, the accuracy increases as we increase the dimensions, this is because uses dimension reduction to achieve greater simplicity and maintainability features.

As for LDA after PCA, the highest accuracy at 30 dimensions and at 20 dimensions. From this result, I did not see a correlation between the number of dimensions with the accuracy. This may be because LDA maximizing the component axes for class-separation, so the increase of dimensions may not always have higher accuracy as such as just PCA.

```
(base) andrewlao@Andrews-MacBook-Air face_recognition % python main.py

10dimensions:
PCA Correct Predictions:177 Wrong Predictions:23 Accuracy:86.5%
LDA Correct Predictions:177  Wrong Predictions:23 Accuracy:88.5%
PCA Confusion Matrix:
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
LDA Confusion Matrix:
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
PCA report:
              precision    recall  f1-score   support

           1       1.00      0.80      0.89         5
           2       0.62      1.00      0.77         5
           3       1.00      1.00      1.00         5
           4       0.83      1.00      0.91         5
           5       1.00      0.80      0.89         5
           6       1.00      1.00      1.00         5
           7       1.00      1.00      1.00         5
           8       1.00      1.00      1.00         5
           9       1.00      1.00      1.00         5
          10       1.00      0.80      0.89         5
          11       0.71      1.00      0.83         5
          12       1.00      1.00      1.00         5
          13       1.00      1.00      1.00         5
          14       1.00      0.20      0.33         5
          15       0.83      1.00      0.91         5
          16       0.83      1.00      0.91         5
          17       0.00      0.00      0.00         5
          18       0.83      1.00      0.91         5
          19       1.00      0.80      0.89         5
          20       1.00      1.00      1.00         5
          21       1.00      1.00      1.00         5
          22       1.00      1.00      1.00         5
          23       1.00      1.00      1.00         5
          24       0.83      1.00      0.91         5
          25       0.71      1.00      0.83         5
          26       1.00      1.00      1.00         5
          27       1.00      0.20      0.33         5
          28       0.50      0.60      0.55         5
          29       1.00      1.00      1.00         5
          30       1.00      1.00      1.00         5
          31       1.00      1.00      1.00         5
          32       1.00      0.40      0.57         5
          33       1.00      1.00      1.00         5
          34       1.00      1.00      1.00         5
          35       1.00      0.40      0.57         5
          36       0.38      0.60      0.46         5
          37       0.83      1.00      0.91         5
          38       0.83      1.00      0.91         5
          39       1.00      1.00      1.00         5
          40       1.00      1.00      1.00         5

    accuracy                           0.86       200
   macro avg       0.89      0.86      0.85       200
weighted avg       0.89      0.86      0.85       200

LDA report:
              precision    recall  f1-score   support

           1       1.00      0.80      0.89         5
           2       1.00      1.00      1.00         5
           3       1.00      1.00      1.00         5
           4       1.00      1.00      1.00         5
           5       1.00      1.00      1.00         5
           6       1.00      1.00      1.00         5
           7       0.83      1.00      0.91         5
           8       0.83      1.00      0.91         5
           9       1.00      1.00      1.00         5
          10       1.00      0.60      0.75         5
          11       0.83      1.00      0.91         5
          12       1.00      1.00      1.00         5
          13       1.00      1.00      1.00         5
          14       1.00      0.20      0.33         5
          15       0.83      1.00      0.91         5
          16       0.83      1.00      0.91         5
          17       0.00      0.00      0.00         5
          18       1.00      1.00      1.00         5
          19       1.00      1.00      1.00         5
          20       1.00      0.80      0.89         5
          21       1.00      1.00      1.00         5
          22       1.00      1.00      1.00         5
          23       1.00      1.00      1.00         5
          24       1.00      1.00      1.00         5
          25       0.83      1.00      0.91         5
          26       1.00      0.80      0.89         5
          27       1.00      0.40      0.57         5
          28       0.43      0.60      0.50         5
          29       0.83      1.00      0.91         5
          30       1.00      1.00      1.00         5
          31       1.00      1.00      1.00         5
          32       1.00      0.60      0.75         5
          33       1.00      1.00      1.00         5
          34       1.00      1.00      1.00         5
          35       1.00      0.80      0.89         5
          36       0.50      0.80      0.62         5
          37       0.83      1.00      0.91         5
          38       0.83      1.00      0.91         5
          39       1.00      1.00      1.00         5
          40       1.00      1.00      1.00         5

    accuracy                           0.89       200
   macro avg       0.91      0.89      0.88       200
weighted avg       0.91      0.89      0.88       200


20dimensions:
PCA Correct Predictions:173 Wrong Predictions:27 Accuracy:87.0%
LDA Correct Predictions:173  Wrong Predictions:27 Accuracy:86.5%
PCA Confusion Matrix:
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
LDA Confusion Matrix:
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 5]]
PCA report:
              precision    recall  f1-score   support

           1       0.83      1.00      0.91         5
           2       0.71      1.00      0.83         5
           3       1.00      0.80      0.89         5
           4       1.00      1.00      1.00         5
           5       1.00      0.80      0.89         5
           6       1.00      1.00      1.00         5
           7       1.00      1.00      1.00         5
           8       0.83      1.00      0.91         5
           9       1.00      1.00      1.00         5
          10       1.00      0.80      0.89         5
          11       0.71      1.00      0.83         5
          12       1.00      1.00      1.00         5
          13       1.00      1.00      1.00         5
          14       1.00      0.20      0.33         5
          15       1.00      1.00      1.00         5
          16       1.00      0.80      0.89         5
          17       0.00      0.00      0.00         5
          18       0.83      1.00      0.91         5
          19       1.00      0.80      0.89         5
          20       1.00      1.00      1.00         5
          21       0.83      1.00      0.91         5
          22       1.00      1.00      1.00         5
          23       1.00      1.00      1.00         5
          24       0.83      1.00      0.91         5
          25       0.83      1.00      0.91         5
          26       1.00      0.80      0.89         5
          27       1.00      0.40      0.57         5
          28       0.43      0.60      0.50         5
          29       1.00      1.00      1.00         5
          30       1.00      1.00      1.00         5
          31       1.00      0.80      0.89         5
          32       1.00      0.60      0.75         5
          33       1.00      1.00      1.00         5
          34       1.00      1.00      1.00         5
          35       1.00      0.80      0.89         5
          36       0.38      0.60      0.46         5
          37       0.83      1.00      0.91         5
          38       1.00      1.00      1.00         5
          39       1.00      1.00      1.00         5
          40       1.00      1.00      1.00         5

    accuracy                           0.87       200
   macro avg       0.90      0.87      0.87       200
weighted avg       0.90      0.87      0.87       200

LDA report:
              precision    recall  f1-score   support

           1       0.62      1.00      0.77         5
           2       0.83      1.00      0.91         5
           3       1.00      1.00      1.00         5
           4       1.00      1.00      1.00         5
           5       1.00      1.00      1.00         5
           6       1.00      1.00      1.00         5
           7       1.00      1.00      1.00         5
           8       0.83      1.00      0.91         5
           9       1.00      1.00      1.00         5
          10       1.00      0.80      0.89         5
          11       0.80      0.80      0.80         5
          12       1.00      1.00      1.00         5
          13       1.00      1.00      1.00         5
          14       1.00      0.20      0.33         5
          15       1.00      1.00      1.00         5
          16       1.00      0.40      0.57         5
          17       0.00      0.00      0.00         5
          18       1.00      1.00      1.00         5
          19       1.00      0.80      0.89         5
          20       1.00      1.00      1.00         5
          21       1.00      1.00      1.00         5
          22       0.83      1.00      0.91         5
          23       0.62      1.00      0.77         5
          24       1.00      1.00      1.00         5
          25       1.00      1.00      1.00         5
          26       1.00      0.80      0.89         5
          27       1.00      0.20      0.33         5
          28       0.43      0.60      0.50         5
          29       0.83      1.00      0.91         5
          30       0.71      1.00      0.83         5
          31       1.00      1.00      1.00         5
          32       1.00      0.80      0.89         5
          33       1.00      0.80      0.89         5
          34       1.00      1.00      1.00         5
          35       1.00      1.00      1.00         5
          36       0.67      0.80      0.73         5
          37       0.71      1.00      0.83         5
          38       0.83      1.00      0.91         5
          39       1.00      0.60      0.75         5
          40       1.00      1.00      1.00         5

    accuracy                           0.86       200
   macro avg       0.89      0.86      0.86       200
weighted avg       0.89      0.86      0.86       200


30dimensions:
PCA Correct Predictions:176 Wrong Predictions:24 Accuracy:89.0%
LDA Correct Predictions:176  Wrong Predictions:24 Accuracy:88.0%
PCA Confusion Matrix:
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
LDA Confusion Matrix:
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
PCA report:
              precision    recall  f1-score   support

           1       1.00      1.00      1.00         5
           2       0.83      1.00      0.91         5
           3       1.00      1.00      1.00         5
           4       1.00      1.00      1.00         5
           5       1.00      0.80      0.89         5
           6       1.00      1.00      1.00         5
           7       1.00      1.00      1.00         5
           8       0.83      1.00      0.91         5
           9       1.00      1.00      1.00         5
          10       1.00      0.80      0.89         5
          11       0.71      1.00      0.83         5
          12       1.00      1.00      1.00         5
          13       1.00      1.00      1.00         5
          14       1.00      0.20      0.33         5
          15       0.83      1.00      0.91         5
          16       1.00      1.00      1.00         5
          17       0.00      0.00      0.00         5
          18       0.83      1.00      0.91         5
          19       1.00      0.80      0.89         5
          20       1.00      1.00      1.00         5
          21       1.00      1.00      1.00         5
          22       1.00      1.00      1.00         5
          23       1.00      1.00      1.00         5
          24       0.83      1.00      0.91         5
          25       1.00      1.00      1.00         5
          26       1.00      0.80      0.89         5
          27       1.00      0.40      0.57         5
          28       0.43      0.60      0.50         5
          29       1.00      1.00      1.00         5
          30       1.00      1.00      1.00         5
          31       1.00      1.00      1.00         5
          32       1.00      0.60      0.75         5
          33       1.00      1.00      1.00         5
          34       1.00      1.00      1.00         5
          35       1.00      1.00      1.00         5
          36       0.38      0.60      0.46         5
          37       0.83      1.00      0.91         5
          38       1.00      1.00      1.00         5
          39       1.00      1.00      1.00         5
          40       1.00      1.00      1.00         5

    accuracy                           0.89       200
   macro avg       0.91      0.89      0.89       200
weighted avg       0.91      0.89      0.89       200

LDA report:
              precision    recall  f1-score   support

           1       0.80      0.80      0.80         5
           2       1.00      1.00      1.00         5
           3       1.00      1.00      1.00         5
           4       1.00      1.00      1.00         5
           5       1.00      1.00      1.00         5
           6       1.00      1.00      1.00         5
           7       1.00      0.80      0.89         5
           8       1.00      1.00      1.00         5
           9       1.00      1.00      1.00         5
          10       1.00      1.00      1.00         5
          11       1.00      0.80      0.89         5
          12       1.00      1.00      1.00         5
          13       1.00      1.00      1.00         5
          14       1.00      0.20      0.33         5
          15       1.00      1.00      1.00         5
          16       1.00      0.80      0.89         5
          17       0.00      0.00      0.00         5
          18       1.00      1.00      1.00         5
          19       1.00      0.80      0.89         5
          20       1.00      0.80      0.89         5
          21       0.71      1.00      0.83         5
          22       0.83      1.00      0.91         5
          23       0.83      1.00      0.91         5
          24       1.00      1.00      1.00         5
          25       1.00      1.00      1.00         5
          26       1.00      1.00      1.00         5
          27       1.00      0.20      0.33         5
          28       0.57      0.80      0.67         5
          29       1.00      1.00      1.00         5
          30       1.00      1.00      1.00         5
          31       1.00      0.80      0.89         5
          32       1.00      0.80      0.89         5
          33       1.00      0.80      0.89         5
          34       1.00      1.00      1.00         5
          35       0.83      1.00      0.91         5
          36       0.50      0.80      0.62         5
          37       0.83      1.00      0.91         5
          38       0.83      1.00      0.91         5
          39       1.00      1.00      1.00         5
          40       0.83      1.00      0.91         5

    accuracy                           0.88       200
   macro avg       0.91      0.88      0.88       200
weighted avg       0.91      0.88      0.88       200


40dimensions:
PCA Correct Predictions:178 Wrong Predictions:22 Accuracy:89.0%
PCA Confusion Matrix:
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
PCA report:
              precision    recall  f1-score   support

           1       1.00      1.00      1.00         5
           2       1.00      1.00      1.00         5
           3       1.00      1.00      1.00         5
           4       1.00      1.00      1.00         5
           5       1.00      0.80      0.89         5
           6       1.00      1.00      1.00         5
           7       1.00      1.00      1.00         5
           8       0.83      1.00      0.91         5
           9       1.00      1.00      1.00         5
          10       1.00      0.80      0.89         5
          11       0.71      1.00      0.83         5
          12       1.00      1.00      1.00         5
          13       1.00      1.00      1.00         5
          14       1.00      0.20      0.33         5
          15       0.83      1.00      0.91         5
          16       1.00      1.00      1.00         5
          17       0.00      0.00      0.00         5
          18       0.83      1.00      0.91         5
          19       1.00      0.80      0.89         5
          20       1.00      1.00      1.00         5
          21       1.00      1.00      1.00         5
          22       1.00      1.00      1.00         5
          23       1.00      1.00      1.00         5
          24       0.83      1.00      0.91         5
          25       0.83      1.00      0.91         5
          26       1.00      0.80      0.89         5
          27       1.00      0.40      0.57         5
          28       0.43      0.60      0.50         5
          29       1.00      1.00      1.00         5
          30       1.00      1.00      1.00         5
          31       1.00      1.00      1.00         5
          32       1.00      0.80      0.89         5
          33       1.00      1.00      1.00         5
          34       1.00      1.00      1.00         5
          35       1.00      0.80      0.89         5
          36       0.38      0.60      0.46         5
          37       0.83      1.00      0.91         5
          38       1.00      1.00      1.00         5
          39       1.00      1.00      1.00         5
          40       1.00      1.00      1.00         5

    accuracy                           0.89       200
   macro avg       0.91      0.89      0.89       200
weighted avg       0.91      0.89      0.89       200


50dimensions:
PCA Correct Predictions:180 Wrong Predictions:20 Accuracy:90.0%
PCA Confusion Matrix:
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
PCA report:
              precision    recall  f1-score   support

           1       1.00      1.00      1.00         5
           2       1.00      1.00      1.00         5
           3       1.00      1.00      1.00         5
           4       1.00      1.00      1.00         5
           5       1.00      1.00      1.00         5
           6       1.00      1.00      1.00         5
           7       1.00      1.00      1.00         5
           8       0.83      1.00      0.91         5
           9       1.00      1.00      1.00         5
          10       1.00      0.80      0.89         5
          11       0.71      1.00      0.83         5
          12       1.00      1.00      1.00         5
          13       1.00      1.00      1.00         5
          14       1.00      0.20      0.33         5
          15       0.83      1.00      0.91         5
          16       1.00      1.00      1.00         5
          17       0.00      0.00      0.00         5
          18       1.00      1.00      1.00         5
          19       1.00      0.80      0.89         5
          20       1.00      1.00      1.00         5
          21       1.00      1.00      1.00         5
          22       1.00      1.00      1.00         5
          23       1.00      1.00      1.00         5
          24       0.83      1.00      0.91         5
          25       1.00      1.00      1.00         5
          26       1.00      0.80      0.89         5
          27       1.00      0.40      0.57         5
          28       0.43      0.60      0.50         5
          29       1.00      1.00      1.00         5
          30       1.00      1.00      1.00         5
          31       1.00      1.00      1.00         5
          32       1.00      0.80      0.89         5
          33       1.00      1.00      1.00         5
          34       1.00      1.00      1.00         5
          35       1.00      1.00      1.00         5
          36       0.38      0.60      0.46         5
          37       0.83      1.00      0.91         5
          38       1.00      1.00      1.00         5
          39       1.00      1.00      1.00         5
          40       1.00      1.00      1.00         5

    accuracy                           0.90       200
   macro avg       0.92      0.90      0.90       200
weighted avg       0.92      0.90      0.90       200
```

## 4. Discussion
For this assignment, one of the issue I faced learning how to **preprocess the data**, when I first saw that the format of the images are not what I am familiar with, I got lost. Thankfully my labmate's gave a couple advice on how to process it. This skill can be improved by practicing more on how to process different data in **Python**.

Next issue is getting a **higher accuracy**, the highest accuracy I got from this is 90.5%, but I would like to get accuracy to be above 95%. I tried to adjust the parameters with GridSearchCV() for SVC, but I did not see greater improvements, thus I left it out of the code. How I can improve it is to do more **research** on ways to **optimize the parameters** to get better results. Since it doesn't take long for this program to run, I can also write a code to **automate** the process of finding the best parameters.

## 5. Summary
In this assignment, I learned different methods on how to **preprocess data** for it to fit the model, and implementing the **PCA** and **LDA** method to recognize face images.

I did feel a little bit of pressure in completing this assignment because I had other final projects. But because the **time for the model to run is significantly shorter** than the last assignment, I was able to complete the assignment in a shorter period of time.

This assignment made me interested in the science behind the parameters of the model, and I want to learn how to **optimize the parameters** to get a higher accuracy.

In all, it was a very interesting and helpful experience, I'm really glad that I had the opportunity to have **hands-on practical experience** with the things we learn and not only just learned about the theory.
