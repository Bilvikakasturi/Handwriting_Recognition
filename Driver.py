#imports
import numpy as nmPy                #data manipulation
import tensorflow as tenFlw         #Neural Network
import pandas as panda              #csv stuff
import matplotlib.pyplot as plt     #backend to write results to file
import cv2                          #openCV for images
import os
import keras
import keras.models as mdl
import keras.layers as lyr


#--LOAD DATA--#

#import the csvs to a panda dataframe
trainSet = pd.read_csv('\written_name_train_v2.csv')
validSet = pd.read_csv('\written_name_validation_v2.csv')
#note: back slashes are Windows paths, if using linux or mac, will need to use forward slash


#--CLEAN DATA--#

##check for missing values in CSV
##if want to know how many null values for statistics reasons, can print values before redefining dataframes
if (trainSet['IDENTITY'].isnull().sum() > 0):
    #drop rows with missing values
    trainSet = trainSet.dropna(axis=0)
if (validSet['IDENTITY'0.isnull().sum() > 0):
    validSet = validSet.dropna(axis=0)


##if you open the csvs, you can see within top ten of validation set that one of the IDENTITY values is UNREADABLE
##these need to be removed. Looking online, you can find how to drop rows if val==something

##we will make copy of trainSet where IDENTITY VAL != UNREADABLE
##but, we want to make sure all are correct case, as to not miss any
trainSet['IDENTITY'] = trainSet['IDENTITY'].str.upper()
validSet['IDENTITY'] = validSet['IDENTITY'].str.upper()


##now we can remove 'unreadable' values by making copy of dataframe without those rows
trainSet = trainSet[trainSet['IDENTITY'] != 'UNREADABLE']
validSet = validSet[validSet['IDENTITY'] != 'UNREADABLE']


##scrolling through the csv, we can ALSO see that some of the ID values are labeled NONE
##if you check the associated image file, you can see those images are blank
##so lets remove those as well
trainSet = trainSet[trainSet['IDENTITY'] != 'NONE']
validSet = validSet[validSet['IDENTITY'] != 'NONE']


##now we reset the index to make up for the drop rows
##drop=True drops the original index from the columns, and replaces it with reset index
trainSet = trainSet.reset_index(drop=True)
validSet = validSet.reset_index(drop=True)


#--Img Processing--#

#make all same size


