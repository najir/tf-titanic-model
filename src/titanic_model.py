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

ageXgender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)

featureColumn = []
derivedFeatureColumn = [ageXgender]

#Iterate on list of columns, append feature columns for each one in lists. Used for estimators
for featureName in CAT_COLUMNS:
    vocab = dfTrain[featureName].unique()
    featureColumn.append(tf.feature_column.categrorical_column_with_vocabulary_lists(featureName, vocab))
for featureName in NUM_COLUMNS:
    featureColumn.append(tf.feature_column.numeric_column(featureName, dtype=tf.float32))


#Creates inputs for our training data and evaluation
trainInput = MakeInputFunction(dfTrain, y_train)
evalInput = MakeInputFunction(dfEval, y_eval, 1, False)

#Create the actual estimator, passing in our feature columns to the linear classification
linearEstim = tf.estimator.LinearClassification(feature_columns=featureColumn+derivedFeatureColumn)

#Train the given model using out training inputs, then evaluate it with out eval inputs
linearEstim.train(trainInput)
result = linearEstim.evaluate(evalInput)

print(result['accuracy'])

evaluatedPredictions = list(linearEstim.predict(evalInput))
probabilities = pd.series(pred['probabilities'][1] for pred in evaluatedPredictions)

probHistogram = probabilities.plot(kind='hist', bins=20, title='Predicted Probabilities')



#Converts our Pandas DataFrame into a usable tensorflow dataframe object
def MakeInputFunction(funcData, funcLabel, epochs=10, shuffle=true, batch=32):
    def InputFunction():
        ds = tf.data.Dataset.from_tensor_slices((dict(funcData), funcLabel))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch).repeat(epochs)
        return ds
    return InputFunction





