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


def main():
    #--LOAD DATA--#

    #import the csvs to a panda dataframe
    trainSet = pd.read_csv('\\written_name_train_v2.csv')
    validSet = pd.read_csv('\\written_name_validation_v2.csv')
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

    #trainRows = trainSet.shape[0]
    #validRows = validSet.shape[0]
    ##loading 330k images into memory is not gunna work. Potentially need to do multiple iterations of multiple batches?

    ##lets start with subset. For now, say train with 10k images maybe? Placeholder value.
    ##can use the xset.shape for statistics though -- total rows after cleaning    


    #now we need to loop through the CSV's to get the images and store in a list
    #during this, we need to read in the image at the dataframe filename,
    #shape into a 64h  256w image
    #then from what I've read, the image needs to be rotated, and pixel intensity set to [0, 1] rather than [0,255]

    trainRange = 5000
    validRange = 1000

    trainImgLst = []
    trainLblLst = []
    validImgLst = []
    validLblLst = []
    ##this will take first trainRange images. If we want to do random, we need to alter a bit
    ##One post I read talked about 'blurring' the image to reduce noise... but I'm not sure
    ##the code would look like: finalImg = cv2GaussianBlur(greyImg, (5, 5),0)
    enumAlpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
        
    for x in range(trainRange):
        imgPath = '\\train\train\\' + trainSet.loc[x, 'FILENAME']
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ##resizes img to 256 w 64 h; there was one notebook that cropped or padded with white space. might be better?
        img = cv2.resize(img, (256, 64))
        ##downscale pixel intensity to range 0-1
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = img/255    
        trainImgLst.append(img)
        label = str(trainSet.loc[x, 'IDENTITY'])
        enumLbl = txt2num(enumAlpha, label)
        trainLblLst.append(enumLbl)

    for y in range(validRange):
        imgPath = '\\train\train\\' + validSet.loc[x, 'FILENAME']
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 64))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = img/255
        validImgLst.append(img)
        label = str(trainSet.loc[x, 'IDENTITY'])
        enumLbl = txt2num(enumAlpha, label)
        validLblLst.append(enumLbl)

    trainImgLst = np.array(trainImgLst).reshape(-1, 256, 64, 1)
    validImgLst = np.array(validImgLst).reshape(-1, 256, 64, 1)    

##generic python enumeration functions for labels; easier than dealing with ord or anything
def txt2num(alphaStr, lblStr):

    nums = []
    for char in lblStr:
        a = alphaStr.find(char)
        if (a>-1):
            nums.append(a)
    return nums

def num2txt(alphaStr, numLst):

    txt = ""
    for num in numLst:
        if num == -1:
            break
        else:
            txt = txt + alphaStr[num]
    return txt


enumLabel=[]
#--CTC Loss--#

#--Model--#

#--Training--#

#--Validation--#

#--Test--#

