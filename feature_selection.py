#Import libraries
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
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif, SelectPercentile

selector = SelectPercentile(f_classif, percentile=10)
fit1=selector.fit(datafeatures,datalabels)
x_train_selected3 = fit.transform(datafeatures)
X_indices = np.arange(datafeatures.shape[-1])
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices - .45, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)', color='blue', edgecolor='black')

#Data normalization
datafeatures = pd.read_csv(io.BytesIO(uploaded['data.csv']))
datalabels = pd.DataFrame([1 if each == 'M' else 0 for each in datafeatures.diagnosis], columns=["label"])
datafeatures.drop(['id','diagnosis','Unnamed: 32'], axis = 1 ,inplace=True)
datafeatures = (datafeatures - datafeatures.min()) / (datafeatures.max() - datafeatures.min())
datafeatures.dropna(axis=1, how='any')

#PCA - 2D and 3D
pca2D = PCA(n_components = 2, whiten = True)
pca2D.fit(datafeatures)
npFeatures2D = pca2D.transform(datafeatures)
dfFeatures2D = pd.DataFrame(npFeatures2D, columns=['p1', 'p2'])
dfLabels2D = datalabels.copy()

pca3D = PCA(n_components = 3, whiten = True)
pca3D.fit(datafeatures)
npFeatures3D = pca3D.transform(datafeatures)
dfFeatures3D = pd.DataFrame(npFeatures3D, columns=['p1', 'p2', 'p3'])
dfLabels3D = datalabels.copy()

print('Variance Ratio for 3D PCA model is: ' + str(pca3D.explained_variance_ratio_))
print('Variance Sum for 3D PCA model is  : ' + str(sum(pca3D.explained_variance_ratio_)))
print('Variance Ratio for 2D PCA model is: ' + str(pca2D.explained_variance_ratio_))
print('Variance Sum for 2D PCA model is : ' + str(sum(pca2D.explained_variance_ratio_)))

#Plot for visualization
dfTemp2D = pd.concat([dfFeatures2D, dfLabels2D], axis=1)
plt.figure(figsize=(8,8))

x = dfTemp2D.p1;
y = dfTemp2D.p2;

plt.scatter(x[dfTemp2D.label == 0],y[dfTemp2D.label == 0], color='green')
plt.scatter(x[dfTemp2D.label == 1],y[dfTemp2D.label == 1], color='blue')

plt.xlabel('P1')
plt.ylabel('P2')
plt.title('PCA for 2D model')
plt.grid()


dfTemp3D = pd.concat([dfFeatures3D, dfLabels3D], axis=1)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

x = dfTemp3D.p1;
y = dfTemp3D.p2;
z = dfTemp3D.p3;

ax.scatter(x[dfTemp3D.label == 0], y[dfTemp3D.label == 0], z[dfTemp3D.label == 0], c = 'g', marker = 'o', s=30)
ax.scatter(x[dfTemp3D.label == 1], y[dfTemp3D.label == 1], z[dfTemp3D.label == 1], c = 'b', marker = 'o', s=30)
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_zlabel('p3')
plt.title('PCA for 3D model')
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif, SelectPercentile

selector = SelectPercentile(f_classif, percentile=10)
fit1=selector.fit(datafeatures,datalabels)
x_train_selected3 = fit.transform(datafeatures)
X_indices = np.arange(datafeatures.shape[-1])
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices - .45, scores, width=.2,label=r'Univariate score ($-Log(p_{value})$)', color='blue',edgecolor='black')

bestfeatures = SelectKBest(score_func=chi2, k=15)
fit = bestfeatures.fit(datafeatures,datalabels)
x_train_selected2 = fit.transform(datafeatures)
df_scores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(datafeatures.columns)
fScores = pd.concat([dfcolumns,df_scores],axis=1)
fScores.columns = ['Features','Score']
dfFeatures15B = datafeatures.iloc[:,[0, 2, 3, 5, 6, 7, 10, 12, 13, 20, 22, 23, 25, 26, 27]].copy()
dfLabels15B = datalabels.copy()
fScores.nlargest(15,'Score')

#Visualization
f, ax = plt.subplots(figsize = (15, 15))
fScores = fScores.nlargest(15,'Score')
fScores.index = fScores.Features
fScores.plot(kind='barh', ax=ax, color="green")

plt.title("Best 15 features")
plt.ylabel("Features")
plt.xlabel("Scores")
plt.grid()



