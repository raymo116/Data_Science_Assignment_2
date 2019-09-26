#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# ---

# # Background
# 
# I had to do this twice because my first dataset didn't work very well. I started with transportation data from the US census, and I switched to the iris dataset just to make sure that I had accurately built a KNN algorithm. I discuss that later on in the writeup.
# 
# 1. Describe the problem (4 points) :
#     1. Describe the dataset
#         - US Census Data
#             - This dataset comes from the 2016 US Census. I've taken a subset of a dataset titled "Means of Transportation" in order to simplify my analysis. Each row in this dataset represents a zip code. From this dataset, I'm using all columns that refer to modes of transportation (for the entire populace, not sectioned by income level), the number of people living under the poverty line, and the number of people surveyed for each zip code. I then used the number of people surveyed to scale each element in the dataset between 0 and 1 (as a percentage of the given populace, divided by 100). I'm using the percentages of modes of transportation as traits and the percentage of people living under the poverty line (rounded to the nearest 5%) as tags.
#         - Iris
#             - This dataset is a common dataset used to practice data science on. It's composed of measurements from three different types of iris as traits - petal length, petal width, sepal length, and sepal width - and tagged by species.
#         
#     2. What the features represent
#         - US Census Data
#             - Each row is a zip code, and it contains the number of people for that zip code that fit into each of the given categories.
#             - Each column is a category of transportation that people can fit into. There are 6 main categories: driving alone, carpooling, public transportation, walking, unconventional means, and working from home. These are each split into three lower levels: less than 100% of the poverty level, 100-149% of the poverty level and at least 150% of the poverty level. There's also a column for the number of people surveyed.
#         - Iris
#             - The petal length, petal width, sepal length, sepal width columns are measurements taken from iris flowers (cm), and the species column is the species. This is divided into I. setosa (0), I. versicolor (1), and I. virginica (2).
#         
#     3. What the target variable is (trying to predict)
#         - US Census Data
#             - Through this analysis, I'm hoping that I will be able to predict the percentage of a zip code that lives below the poverty line based on the popularity of different modes of transportation. This variable is not explicitly given, but is calculated based on several other columns. If desired, however, this data could be converted back to the number of people simply by multiplying the result (on a 0-1 scale) by the number of people surveyed in a zip code.
#         - Iris
#             - For this dataset, I'm hoping to be able to determine correctly choose an iris subspecies given data for all of its traits.
#         
# 2. Split into train and test sets (2 point)
#     - 70% for training, 30% for testing. For the census data, I split it in excel, and for the iris dataset, I split it inside python.
# 
# 3. Implement KNN from scratch (8 points)
#     1. Include plots showing quality metrics for varying numbers of k
#         - Included below
#     2. Recommend a value for k
#         - US Census Data
#             - $k=1$
#         - Iris
#             - $k = 9$
#     3. Why did you choose this value?
#         - US Census
#             - Because that is the point where the prediction is most accurate
#         - Iris
#             - Because any k larger than this is extremely accurate (up to 100%), and therefore in danger of overfitting
# 
# 4. Train a model using Scikit learn KNN (6 points)
#     1. Compare the results to part 1
#         - US Census Data
#             - My model and SKLearn's model had about the same accuracy, but mine was a lot slower (I think because mine was implemented in python, while theirs was likely a python wrapper for a C/C++ implementation. Our models were most accurate at different values of k, but they both had the same maximum accuracy of 57%.
#         - Iris
#             - The same as above, they were almost the same, but since the dataset was smaller here, my algorithm didn't take too long. Optimum ks varied, but they both had the same maximum accuracy of 100%.
#     2. Include plots showing quality metrics for varying numbers of k
#         - Included below
#     3. Recommend a value for k
#         - US Census Data
#             - $k = 11$
#         - Iris
#             - $k = 8$
#     4. Why did you choose this value?
#         - US Census
#             - Because that is the point where the prediction is most accurate
#         - Iris
#             - Because any k larger than this is extremely accurate (up to 100%), and therefore in danger of overfitting
# 
# 5. Extra credit
#     1. Perform cross-validation (1 point)
#     2. Visualize decision boundary in two dimensions (if your data is more than 2d only use 2 features) (1 point)
#     3. 5-minute presentation of your findings (5 points)

# # Import Statements

# In[2]:


import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import math
import operator
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import random


# # Global Variables

# In[39]:


# Global variables

# Tests the number of Ks for the custom algorithm
numOfKs = 20

def ceilt(num):
    return int(math.ceil(num / 10.0)) * 10

def floort(num):
    return int(math.floor(num / 10.0)) * 10


# # Setting Up Data

# In[20]:


# Importing the data with Pandas
names = ["numTotal", "bottomTotal", "midTotal", "topTotal", "numCTVAlone", "bottomCTVAlone", "midCTVAlone", 
         "topCTVAlone", "numCTVPool", "bottomCTVPool", "midCTVPool", "topCTVPool", "numPublic", "bottomPublic", 
         "midPublic", "topPublic", "numWalked", "bottomWalked", "midWalked", "topWalked", "numTMBO", 
         "bottomTMBO", "midTMBO", "topTMBO", "numHome", "bottomHome", "midHome", "topHome", "pUnderP"]
data = pd.read_csv("Working_Data_Raw_Training.csv", names=names)
testingData = pd.read_csv("Working_Data_Raw_Testing.csv", names=names)


# In[21]:


# Training data
# Normalize by turning every stat into a percentage of the total number of people asked
data_perc = data[["numCTVAlone", "numCTVPool", "numPublic", "numWalked", "numTMBO", "numHome"]].div(data["numTotal"].iloc[0], axis='columns')
data_perc["pUnderP"] = data["pUnderP"]

# Testing data
data_perc_testing = testingData[["numCTVAlone", "numCTVPool", "numPublic", "numWalked", "numTMBO", "numHome"]].div(data["numTotal"].iloc[0], axis='columns')
data_perc_testing["pUnderP"] = testingData["pUnderP"]

# Data was randomized in excel before importing

# Spliting up the data in a different way for sklearn knn algorithm
training_data_prebuilt_values = data[["numCTVAlone", "numCTVPool", "numPublic", "numWalked", "numTMBO", "numHome"]]
training_data_prebuilt_tags = data[["pUnderP"]]

testing_data_prebuilt_values = testingData[["numCTVAlone", "numCTVPool", "numPublic", "numWalked", "numTMBO", "numHome"]]
testing_data_prebuilt_tags = testingData[["pUnderP"]]


# # Define Functions for Custom KNN Implimentation

# In[22]:


# Sets up distance function
def euclideanDistance(row1, row2):
    distance = 0
    for x in range(len(row1)-1):
        distance += ((row1[x] - row2[x]) ** 2)
    return math.sqrt(distance)


# In[23]:


# Finds the nearest neighbor
def getNeighbors(source, test, k, fSlice, eSlice):
    smallerSource = source.loc[fSlice:eSlice]
    smallerTest = test.loc[fSlice:eSlice]
    # Array of distances
    distances = []
    # cycle through all for the rows in the source
    for x in range(len(source.index)):
        # compute the distance to that element
        dist = euclideanDistance(test, source.loc[:,fSlice:eSlice].iloc[x])
        distances.append((source.iloc[x,:], dist))
    # sort the distances to get the closest ones
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    # Select the closest neighbors
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# In[24]:


# Choose the nearest neighbor (chooses whichever made it to the front in the case of a tie)
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return float(sortedVotes[0][0])


# In[25]:


# Run my custom KNN algorithm on one item
def runKNNOnSingle(source, test, k, fSlice, eSlice):
    neighbors = getNeighbors(source, test, k, fSlice, eSlice)
    return getResponse(neighbors)
        


# In[26]:


# Run the algorithm on every row in an input set
def testData(trainingSet, testingInput, k, fSlice, eSlice, lSlice):
    KNNresult = runKNNOnSingle(trainingSet, testingInput, k, fSlice, eSlice)
    if (KNNresult == testingInput.loc[lSlice]):
        return 1;
    else:
        return 0;


# In[27]:


# Calculate the accuracy of the KNN algorithm
def runTest(trainingSet, testingSet, k, fSlice, eSlice, lSlice):
    total = len(testingSet)
    correct = 0
    for i in range(len(testingSet)):
        if(i%10==0):
            print("Testing " + str(i) + "/" + str(len(testingSet)))
        correct += testData(trainingSet, testingSet.iloc[i], k, fSlice, eSlice, lSlice)
    
    return correct/total


# # Implimentation of Custom KNN on Census Data

# ## Run Custom KNN

# In[18]:


# WARNING: THIS IS VERY SLOW AND WILL LIKELY TAKE A LONG TIME

# Custom KNN Algorithm on US Census Data

k_values = [None]*numOfKs

for i in range(1, numOfKs+1):
    k_values[i-1] = runTest(data_perc, data_perc_testing, i, "numCTVAlone", "numHome", "pUnderP")
    print("Completed k=" + str(i))


# ## Graphing Accuracy from Custom KNN

# In[1]:


# Custom KNN Algorithm on US Census Data

# Graph the accuracy of the algorithm
k_x = []
for i in range(numOfKs):
    k_x.append(i+1)
    
k_fig = plt.figure()
k_ax = plt.axes()

k_ax.plot(k_x, [i * 100 for i in k_values])

plt.xlabel("Value of k")
plt.ylabel("% Accuracy")

plt.text(0,51,"The maximum accuracy is " +
      str(round(max(k_values)*100)) + "% at k = " +
      str(k_values.index(max(k_values))+1))

plt.title("Custom KNN (Census Data):\nPercent Accuracy as a Function of K")
plt.show(block=True)


# # Implimentation of SKLearn KNN Algorithm

# ## Run SKLearn KNN

# In[7]:


# SKLearn KNN Algorithm on US Census Data

# Check the accuracy for a number of ks
prebuiltResults = [None]*numOfKs

for i in range(1, numOfKs+1):
    k_test_model = KNeighborsClassifier(n_neighbors=i)

    k_test_model.fit(training_data_prebuilt_values, training_data_prebuilt_tags.values.ravel())

    #Predict Output
    k_test_trainingPredicted= k_test_model.predict(testing_data_prebuilt_values)

    k_test_trainingCorrectlyGuessed = 0
    for j in range(len(k_test_trainingPredicted)):
        if(int(testing_data_prebuilt_tags.iloc[j]) == k_test_trainingPredicted[j]):
            k_test_trainingCorrectlyGuessed+=1
    
    prebuiltResults[i-1] = (k_test_trainingCorrectlyGuessed/len(k_test_trainingPredicted))
    print("Completed k=" + str(i))


# ## Graph SKLearn KNN

# In[9]:


# SKLearn KNN Algorithm on US Census Data

# Graph the information for the sklearn model
k_prebuilt_x = []
for i in range(numOfKs):
    k_prebuilt_x.append(i+1)
    
k_prebuilt_fig = plt.figure()
k_prebuilt_ax = plt.axes()

k_prebuilt_ax.plot(k_prebuilt_x, [i * 100 for i in prebuiltResults])

plt.xlabel("Value of k")
plt.ylabel("% Accuracy")

plt.text(0,50,"The maximum accuracy is " +
      str(round(max(prebuiltResults)*100)) + "% at k = " +
      str(prebuiltResults.index(max(prebuiltResults))+1))

plt.title("SKLearn KNN (Census Data):\nPercent Accuracy as a Function of K");
plt.show(block=True)


# # Correlation Graph

# In[39]:


# We had a very unsatisfactory result from our algorithm, so we're going to create a correlation graph
# to try to see how these things are related, if at all

subnames = ["numCTVAlone", "numCTVPool", "numPublic", "numWalked", "numTMBO", "numHome", "pUnderP"]

correlations = data_perc_testing.corr()

# plot correlation matrix
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,7,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(subnames)
ax.set_yticklabels(subnames)

plt.title("Correlation Graph of Columns in Applicable US Census Transportation Data")
plt.text(0,7,"NOTE: It looks like some have BARELY stronger correlations than others, but nothing significant")


plt.show(block=True)

# By creating a correlation graph, none of these fields reall have any influence on the
# percentage of people living under the poverty line.
# It looks like some have BARELY stronger correlations than others, but nothing significant
#
# This makes intuitive sense, as people of all different social standings can use very
# different modes of transportation. Someone can be poor and have multiple cars if they live in
# the country, and someone can be rich and take public transit if they live in the city.
# In order for this data to be more usefull, it's likely we'd need to know more about people's
# geographic location as well as the urban nature of their community.


# # Calculate Accuracy from Random

# In[45]:


# Random Selection

# Check the accuracy of a random selection
guesses = []

#Predict Output
for k in range(len(testing_data_prebuilt_values)):
    guesses.append(testing_data_prebuilt_tags.iloc[random.randint(0, len(testing_data_prebuilt_values)-1)])

randomCorrectlyGuessed = 0
for j in range(len(guesses)):
    if(int(testing_data_prebuilt_tags.iloc[j]) == int(guesses[j])):
        randomCorrectlyGuessed+=1

randomResult = (randomCorrectlyGuessed/len(testing_data_prebuilt_values))
print("Accuracy of Random Selection:", randomResult)


# # Graph Accuracy Of Custom & SKLearn KNN Models and Random Selection

# In[57]:


# Graph the model vs the random sampling
k_prebuilt_x = []
for i in range(numOfKs):
    k_prebuilt_x.append(i+1)
    
k_prebuilt_fig = plt.figure()
k_prebuilt_ax = plt.axes()

plt.ylim([40, 65])

k_prebuilt_ax.plot(k_prebuilt_x, [i * 100 for i in prebuiltResults], label='SKLearn KNN')
k_prebuilt_ax.plot(k_prebuilt_x, [i * 100 for i in k_values], label="Custom KNN")
k_prebuilt_ax.plot(k_prebuilt_x, [randomResult*100 for i in k_prebuilt_x], label='Random')

plt.legend()
plt.xlabel("Value of k")
plt.ylabel("% Accuracy")

plt.text(0,32,"NOTE: This shows that my result is better than random, but it's\n" + 
              "nowhere near useful.  As a result, for this assignment I'm going\n" +
              "to choose a new dataset and build the whole thing from scratch again")

plt.title("All Evaluation Methods(Census Data):\nPercent Accuracy as a Function of K")

plt.show(block=True)


# # Trying Again with Iris Dataset

# ## Import Iris Dataset

# In[10]:


# Import the Iris dataset

tempIris = datasets.load_iris()
irisData = pd.DataFrame({'sLength': tempIris.data[:, 0], 'sWidth': tempIris.data[:, 1],
                     "pLength": tempIris.data[:, 2], "pWidth": tempIris.data[:, 3]})
irisLabels = pd.DataFrame({'Type': tempIris.target})

# Select random ones to perform the algoritm on
msk = np.random.rand(len(irisData)) < 0.8


# Set up for built in KNN
irisTrainingData = irisData[msk]
irisTrainingLables = irisLabels[msk]

irisTestingData = irisData[~msk]
irisTestingLabels = irisLabels[~msk]


# Set up for custom KNN
irisCustomKNNTraining = irisData[msk].copy()
irisCustomKNNTraining["Type"] = irisTrainingLables.copy()

irisCustomKNNTesting = irisData[~msk].copy()
irisCustomKNNTesting["Type"] = irisTestingLabels.copy()


# ## Run SKLearn KNN of Iris 

# In[28]:


# Iris SKLearn NN

# Check the accuracy for a number of ks
prebuiltResults = [None]*numOfKs

for i in range(1, numOfKs+1):
    iris_knn_model = KNeighborsClassifier(n_neighbors=i)

    # Train the model using the training sets
    iris_knn_model.fit(irisTrainingData, irisTrainingLables.values.ravel())

    #Predict Output
    trainingPredicted= iris_knn_model.predict(irisTestingData)

    # Checks to make sure the the algorhtm even works on itself
    trainingCorrectlyGuessed = 0
    for j in range(len(trainingPredicted)):
        if(int(irisTestingLabels.iloc[j]) == trainingPredicted[j]):
            trainingCorrectlyGuessed+=1

    prebuiltResults[i-1] = (trainingCorrectlyGuessed/len(trainingPredicted))

    print("Completed k=" + str(i))


# ## Graph Accuracy of SKLearn Algortithm

# In[29]:


# Iris SKLearn KNN

k_prebuilt_x = []
for i in range(numOfKs):
    k_prebuilt_x.append(i+1)
    
k_prebuilt_fig = plt.figure()
k_prebuilt_ax = plt.axes()

k_prebuilt_ax.plot(k_prebuilt_x, [i * 100 for i in prebuiltResults])

plt.xlabel("Value of k")
plt.ylabel("% Accuracy")

plt.title("SKLearn KNN (Iris Data):\nPercent Accuracy as a Function of K");

plt.text(0,round(min(prebuiltResults)*100)-1.5,"The maximum accuracy is " +
      str(round(max(prebuiltResults)*100)) + "% at k = " +
      str(prebuiltResults.index(max(prebuiltResults))+1))

plt.show(block=True)


# ## Impliment Custom KNN Algorithm

# In[30]:


# Iris Custom KNN

# WARNING: THIS IS VERY SLOW AND WILL LIKELY TAKE A LONG TIME

k_values = [None]*numOfKs

for i in range(1, numOfKs+1):
    k_values[i-1] = runTest(irisCustomKNNTraining, irisCustomKNNTesting, i, "sLength", "pWidth", "Type")
    print("Completed k=" + str(i))


# ## Graph Custom KNN Algorithm

# In[32]:


# Iris Custom KNN

k_prebuilt_x = []
for i in range(numOfKs):
    k_prebuilt_x.append(i+1)
    
k_prebuilt_fig = plt.figure()
k_prebuilt_ax = plt.axes()

k_prebuilt_ax.plot(k_prebuilt_x, [i * 100 for i in k_values])

plt.xlabel("Value of k")
plt.ylabel("% Accuracy")

plt.title("Custom KNN (Iris Data):\nPercent Accuracy as a Function of K");

plt.text(0,round(min(k_values)*100)-1.5,"The maximum accuracy is " +
      str(round(max(k_values)*100)) + "% at k = " +
      str(k_values.index(max(k_values))+1))

plt.show(block=True)


# In[33]:


# Random Selection

# Check the accuracy of a random selection
guesses = []

#Predict Output
for k in range(len(irisTestingData)):
    guesses.append(irisTestingLabels.iloc[random.randint(0, len(irisTestingData)-1)])

randomCorrectlyGuessed = 0
for j in range(len(guesses)):
    if(int(irisTestingLabels.iloc[j]) == int(guesses[j])):
        randomCorrectlyGuessed+=1

randomResult = (randomCorrectlyGuessed/len(irisTestingData))
print("Accuracy of Random Selection:", randomResult)


# # Conclusion

# In[51]:


# Broken graph code from https://matplotlib.org/examples/pylab_examples/broken_axis.html

# Graph the model vs the random sampling
k_prebuilt_x = []
for i in range(numOfKs):
    k_prebuilt_x.append(i+1)
    
f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

y1 = [i * 100 for i in prebuiltResults]
y2 = [i * 100 for i in k_values]
y3 = [randomResult*100 for i in k_prebuilt_x]

l1 = ax.plot(k_prebuilt_x, y1, label='SKLearn KNN', color="blue")
l2 = ax.plot(k_prebuilt_x, y2, label="Custom KNN", color="green")
l3 = ax2.plot(k_prebuilt_x, y3, label='Random', color="orange")

ax.set_ylim(floort(min((min(y1), min(y2)))), ceilt(max((max(y1), max(y2)))))  # outliers only
ax2.set_ylim(floort(y3[0]), ceilt(y3[0]))  # most of the data

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.spines['top'].set_visible(False)

ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

plt.suptitle("All Evaluation Methods (Iris Data):\nPercent Accuracy as a Function of K", y=1.1);
f.text(0.5, 0, 'Value of k', ha='center')
f.text(0, 0.5, '% Accuracy', va='center', rotation='vertical')

plt.legend( (l1[0], l2[0], l3[0]), ("SKLearn KNN", "Custom KNN", "Random"), loc = (0.67, .1), ncol=1)

plt.show(block=True)


# # Conclusion

# We can see here that my model works when there is a known relationship and ability to classify the given data. My model seemed to do a very poor job classifying the data for the US Census, barely above random, and my model did do well here, so I assume that it's an issue with my data. If I had to guess, I would say that it was probably an issue of not having the right properties for the given task. If I wanted to make very specific predictions about a location's poverty, I would have to take the geography, urban environment, and cost of living into account. A farmer could potentially live below the poverty line and still have multiple trucks because they're government-subsidized, while a rich person in NY may decide that having a car is too much hassle. Additionally, someone with a decent wage in a large city can still be living extremely modestly because of the high cost of living, while someone in the country would probably pay a lot less for their living space.
# 
# My conclusion is that the data is incomplete for our current purposes, but that there's nothing inherently wrong with the data.

# In[ ]:




