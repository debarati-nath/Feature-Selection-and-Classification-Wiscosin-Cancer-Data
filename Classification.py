#Import libraries
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.fixes import signature
import seaborn as sns
import warnings
import torch.nn as nn
import torch
import os
import io
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve
# importing the model.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import math
import numpy as np

#Train and test split
dfTrainFeatures, dfTestFeatures, dfTrainLabels, dfTestLabels = train_test_split(datafeatures, datalabels, test_size = 0.2, random_state = 42)
dfTrainFeatures, dfValFeatures, dfTrainLabels, dfVallabels = train_test_split(dfTrainFeatures, dfTrainLabels, test_size=0.2, random_state=42 )
dfTrainFeatures3D, dfTestFeatures3D, dfTrainLabels3D, dfTestLabels3D = train_test_split(dfFeatures3D, dfLabels3D, test_size = 0.2, random_state = 42)
dfTrainFeatures2D, dfTestFeatures2D, dfTrainLabels2D, dfTestLabels2D = train_test_split(dfFeatures2D, dfLabels2D, test_size = 0.2, random_state = 42)
dfTrainFeatures15B, dfTestFeatures15B, dfTrainLabels15B, dfTestLabels15B = train_test_split(dfFeatures15B, dfLabels15B, test_size = 0.2, random_state = 42)

#Logistic regression
logistic_regression_model = LogisticRegressionCV(cv=5)
logistic_regression_model.fit(dfTrainFeatures, dfTrainLabels)
acurracy_v = logistic_regression_model.score(dfValFeatures, dfVallabels)
acurracy_lr = logistic_regression_model.score(dfTestFeatures, dfTestLabels)
predictions_lr = logistic_regression_model.predict(dfTestFeatures)
predictions_lr_prob = logistic_regression_model.predict_proba(dfTestFeatures)[:,1]
macro_precision_lr, macro_recall_lr, macro_fscore_lr, _ = precision_recall_fscore_support(dfTestLabels, predictions_lr, average='macro')
micro_precision_lr, micro_recall_lr, micro_fscore_lr, _ = precision_recall_fscore_support(dfTestLabels, predictions_lr, average='micro')

logistic_regression_model_3D = LogisticRegressionCV(cv=5)
logistic_regression_model_3D.fit(dfTrainFeatures3D, dfTrainLabels3D)
acurracy_3D_lr = logistic_regression_model_3D.score(dfTestFeatures3D, dfTestLabels3D)
predictions_lr_3D = logistic_regression_model_3D.predict(dfTestFeatures3D)
predictions_lr_3D_prob = logistic_regression_model_3D.predict_proba(dfTestFeatures3D)[:,1]
macro_precision_lr_3D, macro_recall_lr_3D, macro_fscore_lr_3D, _ = precision_recall_fscore_support(dfTestLabels, predictions_lr_3D, average='macro')
micro_precision_lr_3D, micro_recall_lr_3D, micro_fscore_lr_3D, _ = precision_recall_fscore_support(dfTestLabels, predictions_lr_3D, average='micro')

logistic_regression_model_2D = LogisticRegressionCV(cv=5)
logistic_regression_model_2D.fit(dfTrainFeatures2D, dfTrainLabels2D)
acurracy_2D_lr = logistic_regression_model_2D.score(dfTestFeatures2D, dfTestLabels2D)
predictions_lr_2D = logistic_regression_model_2D.predict(dfTestFeatures2D)
predictions_lr_2D_prob = logistic_regression_model_2D.predict_proba(dfTestFeatures2D)[:,1]
macro_precision_lr_2D, macro_recall_lr_2D, macro_fscore_lr_2D, _ = precision_recall_fscore_support(dfTestLabels, predictions_lr_2D, average='macro')
micro_precision_lr_2D, micro_recall_lr_2D, micro_fscore_lr_2D, _ = precision_recall_fscore_support(dfTestLabels, predictions_lr_2D, average='micro')

logistic_regression_model_15B = LogisticRegressionCV(cv=5)
logistic_regression_model_15B.fit(dfTrainFeatures15B, dfTrainLabels15B)
acurracy_15B_lr = logistic_regression_model_15B.score(dfTestFeatures15B, dfTestLabels15B)
predictions_lr_15B = logistic_regression_model_15B.predict(dfTestFeatures15B)
predictions_lr_15B_prob = logistic_regression_model_15B.predict_proba(dfTestFeatures15B)[:,1]
macro_precision_lr_15B, macro_recall_lr_15B, macro_fscore_lr_15B, _ = precision_recall_fscore_support(dfTestLabels, predictions_lr_15B, average='macro')
micro_precision_lr_15B, micro_recall_lr_15B, micro_fscore_lr_15B, _ = precision_recall_fscore_support(dfTestLabels, predictions_lr_15B, average='micro')

print('')
print('ACURRACY FOR INITIAL TEST DATA    : ' + str(acurracy_lr))
print('ACURRACY FOR INITIAL VAL DATA    : ' + str(acurracy_v))
print('ACURRACY FOR PCA (2DIMENSION): ' + str(acurracy_2D_lr ))
print('ACURRACY FOR PCA (3DIMENSION): ' + str(acurracy_3D_lr ))
print('ACURRACY FOR (15 BEST FEA.)  : ' + str(acurracy_15B_lr))
print('')
print("MACRO PRECISION FOR INITIAL DATA: " + str(macro_precision_lr))
print("MACRO PRECISION PCA (2DIMENSION): " + str(macro_precision_lr_2D))
print("MACRO PRECISION PCA (3DIMENSION): " + str(macro_precision_lr_3D))
print("MACRO PRECISION (15 BEST FEA.)  : " + str(macro_precision_lr_15B))
print('')
print("MACRO RECALL FOR INITIAL DATA: " + str(macro_recall_lr))
print("MACRO RECALL PCA (2DIMENSION): " + str(macro_recall_lr_2D))
print("MACRO RECALL PCA (3DIMENSION): " + str(macro_recall_lr_3D))
print("MACRO RECALL (15 BEST FEA.)  : " + str(macro_recall_lr_15B))
print('')
print("MACRO FSCORE FOR INITIAL DATA: " + str(macro_fscore_lr))
print("MACRO FSCORE PCA (2DIMENSION): " + str(macro_fscore_lr_2D))
print("MACRO FSCORE PCA (3DIMENSION): " + str(macro_fscore_lr_3D))
print("MACRO FSCORE (15 BEST FEA.)  : " + str(macro_fscore_lr_15B))
print('')
print("MICRO PRECISION FOR INITIAL DATA: " + str(micro_precision_lr))
print("MICRO PRECISION PCA (2DIMENSION): " + str(micro_precision_lr_2D))
print("MICRO PRECISION PCA (3DIMENSION): " + str(micro_precision_lr_3D))
print("MICRO PRECISION (15 BEST FEA.)  : " + str(micro_precision_lr_15B))
print('')
print("MICRO RECALL FOR INITIAL DATA: " + str(micro_recall_lr))
print("MICRO RECALL PCA (2DIMENSION): " + str(micro_recall_lr_2D))
print("MICRO RECALL PCA (3DIMENSION): " + str(micro_recall_lr_3D))
print("MICRO RECALL (15 BEST FEA.)  : " + str(micro_recall_lr_15B))
print('')
print("MICRO FSCORE FOR INITIAL DATA: " + str(micro_fscore_lr))
print("MICRO FSCORE PCA (2DIMENSION): " + str(micro_fscore_lr_2D))
print("MICRO FSCORE PCA (3DIMENSION): " + str(micro_fscore_lr_3D))
print("MICRO FSCORE (15 BEST FEA.)  : " + str(micro_fscore_lr_15B))
print('')

#ROC curve
falsePositiveRate_lr,truePositiveRate_lr,thresholds_lr=roc_curve(dfTestLabels, predictions_lr_prob)
falsePositiveRate_lr_3D,truePositiveRate_lr_3D,thresholds_lr_3D=roc_curve(dfTestLabels, predictions_lr_3D_prob)
falsePositiveRate_lr_2D,truePositiveRate_lr_2D,thresholds_lr_2D=roc_curve(dfTestLabels, predictions_lr_2D_prob)
falsePositiveRate_lr_15B,truePositiveRate_lr_15B,thresholds_lr_15B=roc_curve(dfTestLabels, predictions_lr_15B_prob)

f, ax = plt.subplots(figsize = (6,6))
plt.plot(falsePositiveRate_lr, truePositiveRate_lr, color='red', label="Original Test Train Split")
plt.plot(falsePositiveRate_lr_3D,truePositiveRate_lr_3D, color='green', label="3D PCA")
plt.plot(falsePositiveRate_lr_2D,truePositiveRate_lr_2D, color='blue', label="2D PCA")
plt.plot(falsePositiveRate_lr_15B,truePositiveRate_lr_15B, color='black', label="Top 15 features")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic Regression')
plt.legend()
plt.grid()

# KNN
accurracies_knn = []
accurracies_3D_knn = []
accurracies_2D_knn = []
accurracies_15B_knn = []
mean_squared_error_knn = []
mean_squared_error_2D_knn = []
mean_squared_error_3D_knn = []
mean_squared_error_15B_knn = []

for i in range(100):
    knn_model = KNeighborsClassifier(n_neighbors=i + 1)
    knn_model_3D = KNeighborsClassifier(n_neighbors=i + 1)
    knn_model_2D = KNeighborsClassifier(n_neighbors=i + 1)
    knn_model_15B = KNeighborsClassifier(n_neighbors=i + 1)

    knn_model.fit(dfTrainFeatures, dfTrainLabels)
    knn_model_3D.fit(dfTrainFeatures3D, dfTrainLabels3D)
    knn_model_2D.fit(dfTrainFeatures2D, dfTrainLabels2D)
    knn_model_15B.fit(dfTrainFeatures15B, dfTrainLabels15B)

    pred_knn = knn_model.predict(dfTestFeatures)
    pred_2D_knn = knn_model_2D.predict(dfTestFeatures2D)
    pred_3D_knn = knn_model_3D.predict(dfTestFeatures3D)
    pred_15B_knn = knn_model_15B.predict(dfTestFeatures15B)

    acc = knn_model.score(dfTestFeatures, dfTestLabels)
    acc_3D = knn_model_3D.score(dfTestFeatures3D, dfTestLabels3D)
    acc_2D = knn_model_2D.score(dfTestFeatures2D, dfTestLabels2D)
    acc_15B = knn_model_15B.score(dfTestFeatures15B, dfTestLabels15B)

    mse = mean_squared_error(dfTestLabels, pred_knn, multioutput='raw_values')
    mse_2D = mean_squared_error(dfTestLabels, pred_2D_knn, multioutput='raw_values')
    mse_3D = mean_squared_error(dfTestLabels, pred_3D_knn, multioutput='raw_values')
    mse_15B = mean_squared_error(dfTestLabels, pred_15B_knn, multioutput='raw_values')

    mean_squared_error_knn.append(mse)
    mean_squared_error_2D_knn.append(mse_2D)
    mean_squared_error_3D_knn.append(mse_3D)
    mean_squared_error_15B_knn.append(mse_15B)

    accurracies_knn.append(acc)
    accurracies_3D_knn.append(acc_3D)
    accurracies_2D_knn.append(acc_2D)
    accurracies_15B_knn.append(acc_15B)

print('-ACURRACY FOR DIFFERENT DATA TYPES-')
print('')
print('ACURRACY FOR INITIAL DATA')
print('')

for i in range(100):
    print(str(i + 1) + ' nn acurracy for initial data: ' + str(accurracies_knn[i]))

print('')
print('ACURRACY FOR PCA 3D DATA')
print('')

for i in range(100):
    print(str(i + 1) + ' nn acurracy for pca 3D data: ' + str(accurracies_3D_knn[i]))

print('')
print('ACURRACY FOR PCA 2D DATA')
print('')

for i in range(100):
    print(str(i + 1) + ' nn acurracy for pca 2D data: ' + str(accurracies_2D_knn[i]))

print('')
print('ACURRACY FOR PCA 15B DATA')
print('')

for i in range(100):
    print(str(i + 1) + ' nn acurracy for pca 15B data: ' + str(accurracies_15B_knn[i]))

#Mean squared value
k = [i+1 for i in range(100)]
f, ax = plt.subplots(figsize = (10,10))
plt.plot(k,mean_squared_error_knn,color='red', label="Original")
plt.plot(k,mean_squared_error_2D_knn,color='blue', label="2D PCA")
plt.plot(k,mean_squared_error_3D_knn,color='green', label="3D PCA")
plt.plot(k,mean_squared_error_15B_knn,color='black', label="15 Best")
plt.xlabel('K Values')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error for 4 Different Data Types')
plt.legend()
plt.grid()

average_mse = []
for i in range(100):
    avg = (mean_squared_error_knn[i] + mean_squared_error_2D_knn[i] + mean_squared_error_3D_knn[i] + mean_squared_error_15B_knn[i]) / 4
    average_mse.append(avg)

best_knn = average_mse.index(min(average_mse)) + 1
print('Best K value for K-NN is: ' + str(best_knn))

f, ax = plt.subplots(figsize = (10,10))
plt.plot(average_mse)
plt.xlabel('K Values')
plt.ylabel('Mean Squared Error')
plt.title('Average of the Mean Squared Errors')
plt.grid()

#KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = best_knn)
knn_model.fit(dfTrainFeatures, dfTrainLabels)
acc_knn_val = knn_model.score(dfValFeatures, dfVallabels)
acc_knn = knn_model.score(dfTestFeatures, dfTestLabels)
predictions_knn = knn_model.predict(dfTestFeatures)
predictions_knn_prob = knn_model.predict_proba(dfTestFeatures)[:,1]
macro_precision_knn, macro_recall_knn, macro_fscore_knn, _ = precision_recall_fscore_support(dfTestLabels, predictions_knn, average='macro')
micro_precision_knn, micro_recall_knn, micro_fscore_knn, _ = precision_recall_fscore_support(dfTestLabels, predictions_knn, average='micro')

knn_model_3D = KNeighborsClassifier(n_neighbors = best_knn)
knn_model_3D.fit(dfTrainFeatures3D, dfTrainLabels3D)
acc_knn_3D = knn_model_3D.score(dfTestFeatures3D, dfTestLabels3D)
predictions_knn_3D = knn_model_3D.predict(dfTestFeatures3D)
predictions_knn_3D_prob = knn_model_3D.predict_proba(dfTestFeatures3D)[:,1]
macro_precision_knn_3D, macro_recall_knn_3D, macro_fscore_knn_3D, _ = precision_recall_fscore_support(dfTestLabels, predictions_knn_3D, average='macro')
micro_precision_knn_3D, micro_recall_knn_3D, micro_fscore_knn_3D, _ = precision_recall_fscore_support(dfTestLabels, predictions_knn_3D, average='micro')

knn_model_2D = KNeighborsClassifier(n_neighbors = best_knn)
knn_model_2D.fit(dfTrainFeatures2D, dfTrainLabels2D)
acc_knn_2D = knn_model_2D.score(dfTestFeatures2D, dfTestLabels2D)
predictions_knn_2D = knn_model_2D.predict(dfTestFeatures2D)
predictions_knn_2D_prob = knn_model_2D.predict_proba(dfTestFeatures2D)[:,1]
macro_precision_knn_2D, macro_recall_knn_2D, macro_fscore_knn_2D, _ = precision_recall_fscore_support(dfTestLabels, predictions_knn_2D, average='macro')
micro_precision_knn_2D, micro_recall_knn_2D, micro_fscore_knn_2D, _ = precision_recall_fscore_support(dfTestLabels, predictions_knn_2D, average='micro')

knn_model_15B = KNeighborsClassifier(n_neighbors = best_knn)
knn_model_15B.fit(dfTrainFeatures15B, dfTrainLabels15B)
acc_knn_15B = knn_model_15B.score(dfTestFeatures15B, dfTestLabels15B)
predictions_knn_15B = knn_model_15B.predict(dfTestFeatures15B)
predictions_knn_15B_prob = knn_model_15B.predict_proba(dfTestFeatures15B)[:,1]
macro_precision_knn_15B, macro_recall_knn_15B, macro_fscore_knn_15B, _ = precision_recall_fscore_support(dfTestLabels, predictions_knn_15B, average='macro')
micro_precision_knn_15B, micro_recall_knn_15B, micro_fscore_knn_15B, _ = precision_recall_fscore_support(dfTestLabels, predictions_knn_15B, average='micro')

print('')
print('ACURRACY FOR INITIAL DATA     : ' + str(acc_knn))
print('ACURRACY FOR INITIAL DATA     : ' + str(acc_knn_val))
print('ACURRACY FOR PCA (2DIMENSION) : ' + str(acc_knn_3D))
print('ACURRACY FOR PCA (3DIMENSION) : ' + str(acc_knn_2D))
print('ACURRACY FOR (15 BEST FEAT.)  : ' + str(acc_knn_15B))
print('')
print("MACRO PRECISION FOR INITIAL DATA: " + str(macro_precision_knn))
print("MACRO PRECISION PCA (2DIMENSION): " + str(macro_precision_knn_2D))
print("MACRO PRECISION PCA (3DIMENSION): " + str(macro_precision_knn_3D))
print("MACRO PRECISION (15 BEST FEA.)  : " + str(macro_precision_knn_15B))
print('')
print("MACRO RECALL FOR INITIAL DATA: " + str(macro_recall_knn))
print("MACRO RECALL PCA (2DIMENSION): " + str(macro_recall_knn_2D))
print("MACRO RECALL PCA (3DIMENSION): " + str(macro_recall_knn_3D))
print("MACRO RECALL (15 BEST FEA.)  : " + str(macro_recall_knn_15B))
print('')
print("MACRO FSCORE FOR INITIAL DATA: " + str(macro_fscore_knn))
print("MACRO FSCORE PCA (2DIMENSION): " + str(macro_fscore_knn_2D))
print("MACRO FSCORE PCA (3DIMENSION): " + str(macro_fscore_knn_3D))
print("MACRO FSCORE (15 BEST FEA.)  : " + str(macro_fscore_knn_15B))
print('')
print("MICRO PRECISION FOR INITIAL DATA: " + str(micro_precision_knn))
print("MICRO PRECISION PCA (2DIMENSION): " + str(micro_precision_knn_2D))
print("MICRO PRECISION PCA (3DIMENSION): " + str(micro_precision_knn_3D))
print("MICRO PRECISION (15 BEST FEA.)  : " + str(micro_precision_knn_15B))
print('')
print("MICRO RECALL FOR INITIAL DATA: " + str(micro_recall_knn))
print("MICRO RECALL PCA (2DIMENSION): " + str(micro_recall_knn_2D))
print("MICRO RECALL PCA (3DIMENSION): " + str(micro_recall_knn_3D))
print("MICRO RECALL (15 BEST FEA.)  : " + str(micro_recall_knn_15B))
print('')
print("MICRO FSCORE FOR INITIAL DATA: " + str(micro_fscore_knn))
print("MICRO FSCORE PCA (2DIMENSION): " + str(micro_fscore_knn_2D))
print("MICRO FSCORE PCA (3DIMENSION): " + str(micro_fscore_knn_3D))
print("MICRO FSCORE (15 BEST FEA.)  : " + str(micro_fscore_knn_15B))
print('')

#SVC
svm_model = SVC(random_state = 1)
svm_model.fit(dfTrainFeatures, dfTrainLabels)
acurracy_svm = svm_model.score(dfTestFeatures, dfTestLabels)
acurracy_svm_val = svm_model.score(dfValFeatures, dfVallabels)
predictions_svm = svm_model.predict(dfTestFeatures)
macro_precision_svm, macro_recall_svm, macro_fscore_svm, _ = precision_recall_fscore_support(dfTestLabels, predictions_svm, average='macro')
micro_precision_svm, micro_recall_svm, micro_fscore_svm, _ = precision_recall_fscore_support(dfTestLabels, predictions_svm, average='micro')

svm_model_3D = SVC(random_state = 1)
svm_model_3D.fit(dfTrainFeatures3D, dfTrainLabels3D)
acurracy_3D_svm = svm_model_3D.score(dfTestFeatures3D, dfTestLabels3D)
predictions_svm_3D = svm_model_3D.predict(dfTestFeatures3D)
macro_precision_svm_3D, macro_recall_svm_3D, macro_fscore_svm_3D, _ = precision_recall_fscore_support(dfTestLabels, predictions_svm_3D, average='macro')
micro_precision_svm_3D, micro_recall_svm_3D, micro_fscore_svm_3D, _ = precision_recall_fscore_support(dfTestLabels, predictions_svm_3D, average='micro')

svm_model_2D = SVC(random_state = 1)
svm_model_2D.fit(dfTrainFeatures2D, dfTrainLabels2D)
acurracy_2D_svm = svm_model_2D.score(dfTestFeatures2D, dfTestLabels2D)
predictions_svm_2D = svm_model_2D.predict(dfTestFeatures2D)
macro_precision_svm_2D, macro_recall_svm_2D, macro_fscore_svm_2D, _ = precision_recall_fscore_support(dfTestLabels, predictions_svm_2D, average='macro')
micro_precision_svm_2D, micro_recall_svm_2D, micro_fscore_svm_2D, _ = precision_recall_fscore_support(dfTestLabels, predictions_svm_2D, average='micro')

svm_model_15B = SVC(random_state = 1)
svm_model_15B.fit(dfTrainFeatures15B, dfTrainLabels15B)
acurracy_15B_svm = svm_model_15B.score(dfTestFeatures15B, dfTestLabels15B)
predictions_svm_15B = svm_model_15B.predict(dfTestFeatures15B)
macro_precision_svm_15B, macro_recall_svm_15B, macro_fscore_svm_15B, _ = precision_recall_fscore_support(dfTestLabels, predictions_svm_15B, average='macro')
micro_precision_svm_15B, micro_recall_svm_15B, micro_fscore_svm_15B, _ = precision_recall_fscore_support(dfTestLabels, predictions_svm_15B, average='micro')

#SVC for various gammaValues
gammaValues = np.array([
    math.pow(2, -5),
    math.pow(2, -4),
    math.pow(2, -3),
    math.pow(2, -2),
    math.pow(2, 0),
    math.pow(2, 1)
])

acurracy_svm_arr = []
acurracy_svm_arr_val = []
acurracy_svm2D_arr = []
acurracy_svm3D_arr = []
acurracy_svm15_arr = []

macro_precision_svm_arr = []
macro_precision_svm2D_arr = []
macro_precision_svm3D_arr = []
macro_precision_svm15_arr = []

macro_recall_svm_arr = []
macro_recall_svm2D_arr = []
macro_recall_svm3D_arr = []
macro_recall_svm15_arr = []

micro_precision_svm_arr = []
micro_precision_svm2D_arr = []
micro_precision_svm3D_arr = []
micro_precision_svm15_arr = []

micro_recall_svm_arr = []
micro_recall_svm2D_arr = []
micro_recall_svm3D_arr = []
micro_recall_svm15_arr = []

for x in gammaValues:
    svm_model = SVC(random_state=1, gamma=x)
    svm_model.fit(dfTrainFeatures, dfTrainLabels)
    acurracy_svm = svm_model.score(dfTestFeatures, dfTestLabels)
    acurracy_svm_val = svm_model.score(dfValFeatures, dfVallabels)
    macro_precision_svm, macro_recall_svm, macro_fscore_svm, _ = precision_recall_fscore_support(dfTestLabels,
                                                                                                 svm_model.predict(
                                                                                                     dfTestFeatures),
                                                                                                 average='macro')
    micro_precision_svm, micro_recall_svm, micro_fscore_svm, _ = precision_recall_fscore_support(dfTestLabels,
                                                                                                 svm_model.predict(
                                                                                                     dfTestFeatures),
                                                                                                 average='micro')

    acurracy_svm_arr = np.append(acurracy_svm_arr, acurracy_svm)
    acurracy_svm_arr_val = np.append(acurracy_svm_arr_val, acurracy_svm_val)
    macro_precision_svm_arr = np.append(macro_precision_svm_arr, macro_precision_svm)
    macro_recall_svm_arr = np.append(macro_recall_svm_arr, macro_recall_svm)
    micro_precision_svm_arr = np.append(micro_precision_svm_arr, micro_precision_svm)
    micro_recall_svm_arr = np.append(micro_recall_svm_arr, micro_recall_svm)

    svm_model_3D = SVC(random_state=1, gamma=x)
    svm_model_3D.fit(dfTrainFeatures3D, dfTrainLabels3D)
    acurracy_3D_svm = svm_model_3D.score(dfTestFeatures3D, dfTestLabels3D)
    macro_precision_svm_3D, macro_recall_svm_3D, __, _ = precision_recall_fscore_support(dfTestLabels,
                                                                                         svm_model_3D.predict(
                                                                                             dfTestFeatures3D),
                                                                                         average='macro')
    micro_precision_svm_3D, micro_recall_svm_3D, __, _ = precision_recall_fscore_support(dfTestLabels,
                                                                                         svm_model_3D.predict(
                                                                                             dfTestFeatures3D),
                                                                                         average='micro')

    acurracy_svm3D_arr = np.append(acurracy_svm3D_arr, acurracy_3D_svm)
    macro_precision_svm3D_arr = np.append(macro_precision_svm3D_arr, macro_precision_svm_3D)
    macro_recall_svm3D_arr = np.append(macro_recall_svm3D_arr, macro_recall_svm_3D)
    micro_precision_svm3D_arr = np.append(micro_precision_svm3D_arr, micro_precision_svm_3D)
    micro_recall_svm3D_arr = np.append(micro_recall_svm3D_arr, micro_recall_svm_3D)

    svm_model_2D = SVC(random_state=1, gamma=x)
    svm_model_2D.fit(dfTrainFeatures2D, dfTrainLabels2D)
    acurracy_2D_svm = svm_model_2D.score(dfTestFeatures2D, dfTestLabels2D)
    macro_precision_svm_2D, macro_recall_svm_2D, __, _ = precision_recall_fscore_support(dfTestLabels,
                                                                                         svm_model_2D.predict(
                                                                                             dfTestFeatures2D),
                                                                                         average='macro')
    micro_precision_svm_2D, micro_recall_svm_2D, __, _ = precision_recall_fscore_support(dfTestLabels,
                                                                                         svm_model_2D.predict(
                                                                                             dfTestFeatures2D),
                                                                                         average='micro')

    acurracy_svm2D_arr = np.append(acurracy_svm2D_arr, acurracy_2D_svm)
    macro_precision_svm2D_arr = np.append(macro_precision_svm2D_arr, macro_precision_svm_2D)
    macro_recall_svm2D_arr = np.append(macro_recall_svm2D_arr, macro_recall_svm_2D)
    micro_precision_svm2D_arr = np.append(micro_precision_svm2D_arr, micro_precision_svm_2D)
    micro_recall_svm2D_arr = np.append(micro_recall_svm2D_arr, micro_recall_svm_2D)

    svm_model_15B = SVC(random_state=1, gamma=x)
    svm_model_15B.fit(dfTrainFeatures15B, dfTrainLabels15B)
    acurracy_15B_svm = svm_model_15B.score(dfTestFeatures15B, dfTestLabels15B)
    macro_precision_svm_15B, macro_recall_svm_15B, __, _ = precision_recall_fscore_support(dfTestLabels,
                                                                                           svm_model_15B.predict(
                                                                                               dfTestFeatures15B),
                                                                                           average='macro')
    micro_precision_svm_15B, micro_recall_svm_15B, __, _ = precision_recall_fscore_support(dfTestLabels,
                                                                                           svm_model_15B.predict(
                                                                                               dfTestFeatures15B),
                                                                                           average='micro')

    acurracy_svm15_arr = np.append(acurracy_svm15_arr, acurracy_15B_svm)
    macro_precision_svm15_arr = np.append(macro_precision_svm15_arr, macro_precision_svm_15B)
    macro_recall_svm15_arr = np.append(macro_recall_svm15_arr, macro_recall_svm_15B)
    micro_precision_svm15_arr = np.append(micro_precision_svm15_arr, micro_precision_svm_15B)
    micro_recall_svm15_arr = np.append(micro_recall_svm15_arr, micro_recall_svm_15B)

#Plot
f, ax = plt.subplots(figsize = (7,7))
plt.plot(gammaValues, acurracy_svm_arr, '-.',color='green', linewidth=3, label='raw')
plt.plot(gammaValues, acurracy_svm2D_arr,'-',color='blue', linewidth=3, label='2D PCA')
plt.plot(gammaValues, acurracy_svm3D_arr,':' ,color='red', linewidth=3, label='3D PCA')
plt.plot(gammaValues, acurracy_svm15_arr,'r--.', color='cyan', linewidth=3, label='15 Best')
plt.xlabel('Gamma Values')
plt.ylabel('Accuracies')
plt.legend()
plt.grid()

f, ax = plt.subplots(figsize = (7,7))
plt.plot(gammaValues, micro_recall_svm_arr,'-.',color='green', linewidth=3, label='raw')
plt.plot(gammaValues, macro_recall_svm2D_arr,'-',color='blue', linewidth=3, label='2D PCA')
plt.plot(gammaValues, macro_recall_svm3D_arr,':' ,color='red', linewidth=3, label='3D PCA')
plt.plot(gammaValues, macro_recall_svm15_arr,'r---.', color='cyan', linewidth=3, label='15 Best')
plt.xlabel('Gamma Values')
plt.ylabel('Macro Recall Values')
plt.legend()
plt.grid()

f, ax = plt.subplots(figsize = (7,7))
plt.plot(gammaValues, micro_precision_svm_arr,'-.',color='green', linewidth=3, label='raw')
plt.plot(gammaValues, macro_precision_svm2D_arr,'-',color='blue', linewidth=3, label='2D PCA')
plt.plot(gammaValues, macro_precision_svm3D_arr,':' ,color='red', linewidth=3, label='3D PCA')
plt.plot(gammaValues, macro_precision_svm15_arr,'r---.', color='cyan', linewidth=3, label='15 Best')
plt.xlabel('Gamma Values')
plt.ylabel('Macro Precision Values')
plt.legend()
plt.grid()