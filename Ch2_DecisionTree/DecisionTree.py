from numpy import zeros
import numpy as np
from math import log
import matplotlib.pyplot as plt
import pysnooper



def file2matrix(filename, columnscount, hasHeader=False):
    fr = open(filePath, 'r', encoding='utf-8')
    headerCount = 1 if hasHeader else 0
    alllines = fr.readlines()
    returnMat = []
    classLabelVector = []
    fr = open(filename)
    columns = alllines[0].split('\t') if hasHeader else None
    index = 0
    for lineindex in range(2, len(alllines)):
        line = alllines[lineindex]
        listFromLine = line.split('\t')
        returnMat.append(listFromLine)
        classLabelVector.append(("True" in listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector, columns

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


@pysnooper.snoop()
def chooseBestFeatureToSplitExt(dataSet,columns):
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; 
    bestColumns = -1
    dictInfoGain={}
    for i in range(len(columns)-1):
        featList = [row[i] for row in dataSet]#creat
        distink_Vals = set(featList)
        newEntropy=[]
        sumEntropy=0;
        for val in distink_Vals:
            subDataSet = splitDataSet(dataSet, i, val)
            P=len(subDataSet)/float(len(dataSet))
            thisEntropy=calcShannonEnt(subDataSet)
            newEntropy.append( P * calcShannonEnt(subDataSet))
            sumEntropy+=newEntropy[-1]
        infoGain = baseEntropy - sumEntropy
        dictInfoGain[columns[i]]=infoGain
    print(dictInfoGain)



def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer


filePath = "E:\Document Code\Code Pensonal\MachineLearningGitHub\Ch2_DecisionTree\watermelon.txt"
returnMat, classLabelVector, columns = file2matrix(filePath, 6, True)
res=chooseBestFeatureToSplitExt(returnMat,columns)

