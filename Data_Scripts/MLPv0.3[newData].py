#!/usr/bin/env python
# coding: utf-8

# # 5/12/21
# Machine Learning Group Project
#this file contains data form Dylan Kierans for importing and processing the
#Data set, and code written by Alex Sanfilippo for creating an MLP model
#with relevant cross validation



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", size=14)


# # Checking csv and titles by importing first row of a file

# In[52]:


file_dir="../SpotifyData/"
file_name="ClassicRock_3Ho3iO0iJykgEQNbjB2sic.csv"

test_csv = pd.read_csv(file_dir+file_name, index_col=0, nrows=1) 
#display(test_csv)

file_dir="../SpotifyData/"

file_names= ["JazzClassic_37i9dQZF1DXbITWG1ZJKYt.csv",\
             "CountryHits_1mJhSx6aYQmINsZ8dG4gzU.csv",\
             "ClassicalStudy_6wObnEPQ63a4kei1sEcMdH.csv",\
             "HipHopClassics_5CRJExGSQFYXem2hT5GFkd.csv",\
            "ClassicRock_3Ho3iO0iJykgEQNbjB2sic.csv",\
             "KpopClassics_0jB4ANR4ox65etDMnxvGLp.csv",\
             "HeavyMetalClassics_27gN69ebwiJRtXEboL12Ih.csv",\
             "DanceHits_5oKz4DsTP8zbL97UIPbqp4.csv"]

#file_nrows=[ 200, 205, 227,229, 299, 322, 385, 434]


# tempo_column=11
# mode 5
parameter_columns = [1,2,5,6,7,8,9,10,11]



## Most convenient to set up dataframe with initial file, then systematically loop through the others
## 1 indexing here more natural
playlist_number=1
data = pd.read_csv(file_dir+file_names[playlist_number-1], usecols=parameter_columns)#, nrows=file_nrows[0])

playlist_numbers = playlist_number*np.ones(len(data))
data["playlist_number"]=playlist_numbers #Now has extra playlist column



# Now looping through the remaining filenames
for playlist_number in range(2,len(file_names)+1):
    ## dtmp := Temporary dataframe, will append to df.data once data filtered and prepped
    dtmp= pd.read_csv(file_dir+file_names[playlist_number-1], usecols=parameter_columns)
    
    ## Incase we want to use const number of songs per playlist
    #dtmp= pd.read_csv(file_dir+file_names[playlist_number-1], usecols=parameter_columns), \
    #                  nrows=200)

    playlist_numbers = playlist_number*np.ones(len(dtmp)) # array of playlist_number
    dtmp["playlist_number"]=playlist_numbers #Now has extra playlist column
    data=data.append(dtmp, ignore_index=True) #Append back to main df.data
    

    
## Removing duplicates
print("Removing n duplicates = ", data.duplicated().sum()) #prints number of duplicates
data = data.drop_duplicates() 
#display(data.head())

## Reseting indexing after dropping duplicates
data.reset_index(drop=True, inplace=True)

#X = data.loc[:,"danceability":"tempo"]
X = data.loc[:,"danceability":"tempo"]
Y = data.loc[:,"playlist_number"]
X_norm = data.iloc[:, [0,1,3,4,5,6,7]]#Just 7 normalized_params

#Numpy array more convenient
X=np.array(X);Y=np.array(Y)
X_norm = np.array(X_norm)
classes=np.array([1,2,3,4,5,6,7,8])



###START OF ALEX'S CODE: MLP MODEL
#Last Update: 5 Dec, 2021
#Goal: Consider Cross-Validatino of number of hidden layers and L2
#regularization coefficent "C" to best train a Multi-Layer Perceptron
#(MLP) model on the spotify data.  


#Seperate the features from the targets in the data dataframe
X0=data.iloc[:,0] #assign first input feature to X1
X1=data.iloc[:,1] #assign second input feature to X2, etc...
#X2=data.iloc[:,2] #loudness removed 
X3=data.iloc[:,3]
X4=data.iloc[:,4]
X5=data.iloc[:,5]
X6=data.iloc[:,6]
X7=data.iloc[:,7]
X=np.column_stack((X0,X1,X3,X4,X5,X6,X7)) #put all inputs into one matrix
#X=np.column_stack((X3,X4,X5)) #testing alternate inputs
Y=data.loc[:,"playlist_number"] #assign targets to Y

#Split train/test data 80:20
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.2)

#This dummy classifier will serve as baseline for our models
#It predicts a class at random (uniform random variable)
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="uniform")
dummy.fit(xtrain,ytrain)
dummyOutput = dummy.predict(xtest)

###=============CROSS-VALIDATION: MLP============

#1: Number of Nodes in the Hidden Layers
layerscrossval=False #allows us to switch off without messy commenting-out
if layerscrossval:
    mean_error=[]; std_error=[]
    hidden_layer_range = [5,10,25,50,75,100]
    C = 1
    for n in hidden_layer_range:
        print("hidden layer size %d\n"%n)
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(n), alpha=0, max_iter=10000)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    #Plot the error bar graph
    plt.errorbar(hidden_layer_range,mean_error,yerr=std_error,linewidth=3,label = 'MLP Alpha=0')
    plt.plot([0,100],[.33,.33],label = 'Cosine Simularity Baseline')
    plt.plot([0,100],[.18,.18],label = 'Naive Baseline')
    #plt.title('Cross-Validation of Hidden Layers by Accuracy with C = %d'%C)
    plt.legend(loc="best")
    plt.xlabel('Number of Hidden Layer Nodes'); plt.ylabel('Accuracy')
    plt.show()
#2: Value of L2 regularization coefficient C
Ccrossval = False
if Ccrossval:
    mean_error=[]; std_error=[]
    C_range = [1,5,10,100,1000]
    for Ci in C_range:
        print("C %d\n"%Ci)
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(50), alpha = 1.0/Ci, max_iter=10000)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    #Plot the error bar graph
    plt.errorbar(C_range,mean_error,yerr=std_error,linewidth=3, label = "MLP")
    plt.plot([0,1000],[.33,.33],label = 'Cosine Simularity Baseline')
    plt.plot([0,1000],[.18,.18],label = 'Naive Baseline')
    #plt.title('Cross-Validation of C by Accuracy')
    plt.legend(loc="best")
    plt.xlabel('C'); plt.ylabel('Accuracy')
    plt.show()

#number of hidden layers
nLayerscrossval = True
if nLayerscrossval:
    mean_error=[]; std_error=[]
    layerLayouts = [(50),(25,25),(16,16,16),(12,12,12,12),(10,10,10,10,10)]
    for LL in layerLayouts:
        print("Hidden Layers: ",LL,"\n")
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=LL, alpha = 1.0/1000, max_iter=10000)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring='accuracy')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    #Plot the error bar graph
    plt.errorbar([1,2,3,4,5],mean_error,yerr=std_error,linewidth=3,label='MLP')
    plt.plot([1,5],[.33,.33],label = 'Cosine Simularity Baseline')
    plt.plot([1,5],[.18,.18],label = 'Naive Baseline')
    #plt.title('Cross-Validation of Depth by Accuracy')
    plt.legend(loc="best")
    plt.xlabel('Network Depth'); plt.ylabel('Accuracy')
    plt.show()
        



###Creating the MLP Model
##from sklearn.neural_network import MLPClassifier
##model = MLPClassifier(hidden_layer_sizes=(50), alpha=1.0/1000, solver = 'lbfgs',
##                      tol = 0.0001)
##model.max_iter = 10000 #number of training loops until stop
##model.fit(xtrain, ytrain) #train the model
##
###look at predictions on training and testing data
##trainPreds = model.predict(xtrain)
##mlpPreds = model.predict(xtest)
##
###get the accuracy of the mlp model
##from sklearn.metrics import accuracy_score
##mlpAcc = accuracy_score(ytest, mlpPreds)
##print("MLP Model Accuracy: %f"%mlpAcc)
##
###precision of the mlp Model
##from sklearn.metrics import precision_score
##mlpPrec = precision_score(ytest,mlpPreds, average = 'weighted')
##print("MLP Model Precision: %f"%mlpPrec)
##
###accuracy on the training data
##mlpTrainAcc = accuracy_score(ytrain, trainPreds)
##print("MLP Model Accuracy (Training Data): %f"%mlpTrainAcc)
##
###Create a Confusion Matrix
##from sklearn.metrics import confusion_matrix
##print(confusion_matrix(ytest, mlpPreds))
##
##
###Generate AUC statistic 
##preds = model.predict_proba(xtest)
##print(model.classes_)
##from sklearn.metrics import roc_auc_score
##auc = roc_auc_score(ytest, preds, multi_class = 'ovr', average='macro')
##print("AUC for MLP = %f"%auc)
##







