import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

#Used for feature columns later
CAT_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUM_COLUMNS = ['age', 'fare']

#One dataset is used for testing our data but a fresh set of data is needed for evaluating our model
dfTrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfEval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

#Seperating our given outputs to y and keeping data classifications inside df
y_train = dfTrain.pop('survived')
y_eval = dfEval.pop('survived')

print(dfTrain.head()) #Simply print first 5 values in training dataset

print(dfTrain.describe()) #A general list of statistical info based on the DF

print(dfTrain.shape) #Shape of the matrix/vertex/arrays for our df

#####
# Graphs below need to be output
#####

trainAgeHist = dfTrain.age.hist(bins = 20) #Creates a histogram of DF age values with 20 seperate bins

#Creates a bar graph of counts for male/female passengers
trainSexPlot = dfTrain.sex.value_counts().plot(kind = 'barh')

#Another bar graph showing the amount of passengers inside each class
trainClassPlot = dfTrain['class'].value_counts().plot(kind = "barh")

#Recombines training data and survived values, groups them based on M/F
#Creates the mean value of survived and sex, plots on bar graph, also sets custom label
trainSexMeanPlot = pd.concat([dfTrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% Survived')

featureColumn = []

#Iterate on list of columns, append feature columns for each one in lists. Used for estimators
for featureName in CAT_COLUMNS:
    vocab = dfTrain[featureName].unique()
    featureColumn.append(tf.feature_column.categrorical_column_with_vocabulary_lists(featureName, vocab))
for featureName in NUM_COLUMNS:
    featureColumn.append(tf.feature_column.numeric_column(featureName, dtype=tf.float32))


