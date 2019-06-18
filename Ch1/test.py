from numpy import *


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    print(minVals,maxVals)
    ranges = maxVals - minVals
    print(ranges)
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    print(m)
    normDataSet = dataSet - tile(minVals, (m,1))
    print(normDataSet)
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

datingDataMat,datingLabels=file2matrix("E:\Document Code\Code Pensonal\MachineLearning\machinelearning\Ch1\datingTestSet2.txt")
normMat, ranges, minVals=autoNorm(datingDataMat)


hoRatio = 0.50 
m = datingDataMat.shape[0]
numTestVecs = int(m*hoRatio)
print(numTestVecs)
for i in range(numTestVecs):
    classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
    print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
    if (classifierResult != datingLabels[i]): errorCount += 1.0
print("the total error rate is: %f" % (errorCount/float(numTestVecs)))