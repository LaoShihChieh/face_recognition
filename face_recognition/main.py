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

# load datasets and transform them into 1D array
train_dataset, train_target, test_dataset, test_target = preprocessing.load_datasets()
train_img, test_img, train_target, test_target = preprocessing.transform_to_1D_array(train_dataset, train_target, test_dataset, test_target)

# set the parameters for SVC
pca_clf = SVC(kernel='linear', class_weight='balanced')
lda_clf = SVC(kernel='linear', class_weight='balanced')

# apply the PCA model with 10,20,30,40,50 dimensions
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

    # apply the LDA model with 0 to 39 dimensions
    if(i<40):
        # Because LDA is supervised learning, so we need the train_target in LDA.fit().
        LDA_model = LDA(n_components=i).fit(train_img, train_target)

        # LDA transform
        lda_train_img = LDA_model.transform(train_img)
        lda_test_img = LDA_model.transform(test_img)
        
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
