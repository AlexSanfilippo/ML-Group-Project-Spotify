#!/usr/bin/env python
# coding: utf-8

# # Version 1.0
# # 30/11/21
# Machine Learning Group Project
# 
# 
# Adapted from file "ImportingDataV0.1.py"

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", size=14)


# # Checking csv and titles by importing first row of a file

# In[34]:


file_dir="../SpotifyData/"
file_name="ClassicRock_3Ho3iO0iJykgEQNbjB2sic.csv"

test_csv = pd.read_csv(file_dir+file_name, index_col=0, nrows=1) 
display(test_csv)


# # Compiling Data
# Next step is to make a single dataframe holding the data from all source files

# In[48]:


file_dir="../SpotifyData/"

file_names= ["JazzClassic_37i9dQZF1DXbITWG1ZJKYt.csv",             "CountryHits_1mJhSx6aYQmINsZ8dG4gzU.csv",             "ClassicalStudy_6wObnEPQ63a4kei1sEcMdH.csv",             "HipHopClassics_5CRJExGSQFYXem2hT5GFkd.csv",            "ClassicRock_3Ho3iO0iJykgEQNbjB2sic.csv",             "KpopClassics_0jB4ANR4ox65etDMnxvGLp.csv",             "HeavyMetalClassics_27gN69ebwiJRtXEboL12Ih.csv",             "DanceHits_5oKz4DsTP8zbL97UIPbqp4.csv"]

file_nrows=[ 200, 205, 227,229, 299, 322, 385, 434]

# Most convenient to set up dataframe with initial file, then systematically loop through the others
#Sticking with just the basic parameters for now
parameter_columns = [1,2,5,6,7,8,9,10]
# tempo_column=11
# mode 5



#1 indexing here more natural
playlist_number=1
data = pd.read_csv(file_dir+file_names[playlist_number-1], usecols=parameter_columns, nrows=file_nrows[0])

playlist_numbers = playlist_number*np.ones(len(data))
data["playlist_number"]=playlist_numbers #Now has extra playlist column



# Now looping through the remaining filenames
for playlist_number in range(2,len(file_names)+1):
    # dtmp := Temporary dataframe, will append to df.data once data filtered and prepped
    dtmp= pd.read_csv(file_dir+file_names[playlist_number-1], usecols=parameter_columns,                       nrows=file_nrows[playlist_number-1])

    playlist_numbers = playlist_number*np.ones(len(dtmp)) # array of playlist_number
    dtmp["playlist_number"]=playlist_numbers #Now has extra playlist column
    data=data.append(dtmp, ignore_index=True) #Append back to main df.data
    
display(data.head())

print("Removing n duplicates = ", data.duplicated().sum()) #prints number of duplicates
data = data.drop_duplicates() 


X = data.loc[:,"danceability":"liveness"]#.values
#X = data.loc[:,"danceability":"tempo"]
Y = data.loc[:,"playlist_number"]


X.reset_index(drop=True, inplace=True)
Y.reset_index(drop=True, inplace=True)
X=np.array(X);Y=np.array(Y)
classes=np.array([1,2,3,4,5,6,7,8])


# In[49]:


print(X)
print(X.shape)
print(np.array(X)[0])


# # Training?
# Any pre-processing necessary for:
# - Repeat songs
# - Incomplete data
# 
# 
# Note: I have only chosen 200 songs from each playlist, some do contain more songs
# 

# # kNN Model

# In[50]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3, weights="uniform").fit(X,Y)
ypred = model.predict(X)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ypred,Y))


# # Precision for kNN weighted by distance

# In[51]:


#X = data.loc[:,"danceability":"liveness"].values
#Y = data.loc[:,"playlist_number"]

n_splits = 5
ks=[3,5,7,9]

from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold    
from sklearn.metrics import precision_score, accuracy_score
#from sklearn.metrics import 
kf = KFold(n_splits=n_splits, shuffle=True) #important to shuffle for our ordered data

tmp=[]
precision=[]
precision_err=[]
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    for train, test in kf.split(X):
        model.fit(X[train],Y[train])
        ypred = model.predict(X[test])
        #print(confusion_matrix(ypred,Y[test]))

        #tmp.append( precision_score(Y[test], ypred, average="weighted") )
        tmp.append( accuracy_score(Y[test], ypred) )
        ## average = {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’}

    precision.append( np.array(tmp).mean() )
    precision_err.append( np.array(tmp).std() )

print(precision)
print(precision_err)

plt.errorbar(ks, precision, yerr=precision_err)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Precision with average weighting, kNN distance weighting")
plt.show()


# # Accuracy for distance kNN weights

# In[52]:


prec_tmp=[]
precision=[]
precision_err=[]
score_tmp=[]
score=[]
score_err=[]
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    
    score_tmp=[]
    for train, test in kf.split(X):
        model.fit(X[train],Y[train])
        ypred = model.predict(X[test])
        #print(confusion_matrix(ypred,Y[test]))
        score_tmp.append(model.score(X[test],Y[test]))#accuracy score
    score.append(np.array(score_tmp).mean())
    score_err.append(np.array(score_tmp).std())
    
print(score)
print(score_err)

plt.errorbar(ks, score, yerr=score_err)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Score kNN (weighted by distance)")
plt.show()


# #Skeane77 — Today at 11:35
# 
# file_names= ["ClassicRock_3Ho3iO0iJykgEQNbjB2sic.csv",\
#              "ClassicalStudy_6wObnEPQ63a4kei1sEcMdH.csv",\
#              "CountryHits_1mJhSx6aYQmINsZ8dG4gzU.csv",\
#              "DanceHits_5oKz4DsTP8zbL97UIPbqp4.csv",\
#              "HeavyMetalClassics_27gN69ebwiJRtXEboL12Ih.csv",\
#              "HipHopClassics_5CRJExGSQFYXem2hT5GFkd.csv",\
#              "JazzClassic_37i9dQZF1DXbITWG1ZJKYt.csv",\
#              "KpopClassics_0jB4ANR4ox65etDMnxvGLp.csv"]
# 
# 
# # Most convenient to set up dataframe with initial file, then systematically loop through the others
# #Sticking with just the basic parameters for now
# parameter_columns = [1,2,5,6,7,8,9, 19]#19 for songnames
# nrows=200
# 
# 
# 
# #1 indexing here more natural
# playlist_number=1
# data = pd.read_csv(file_dir+file_names[playlist_number-1], usecols=parameter_columns, nrows=nrows)
# 
# playlist_numbers = playlist_number*np.ones(len(data))
# data["playlist_number"]=playlist_numbers #Now has extra playlist column
# 
# 
# 
# # Now looping through the remaining filenames
# for playlist_number in range(2,len(file_names)+1):
#     # dtmp := Temporary dataframe, will append to df.data once data filtered and prepped
#     dtmp= pd.read_csv(file_dir+file_names[playlist_number-1], usecols=parameter_columns, nrows=nrows)
# 
#     playlist_numbers = playlist_number*np.ones(len(dtmp)) # array of playlist_number
#     dtmp["playlist_number"]=playlist_numbers #Now has extra playlist column
#     data=data.append(dtmp, ignore_index=True) #Append back to main df.data
#     
# display(data)
# print(data.duplicated().sum()) #prints number of duplicates
# print(data.loc[data.duplicated(),:]) #prints rowof duplicate

# # Polynomial Features
# 

# In[53]:


from sklearn.preprocessing import PolynomialFeatures
q_poly = [1,2,3,4,5]      

for q in q_poly:
    Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X)
    tmp=[]
    precision=[]
    precision_err=[]
    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train],Y[train])
            ypred = model.predict(Xpoly[test])
            #print(confusion_matrix(ypred,Y[test]))

            tmp.append( precision_score(Y[test], ypred, average="weighted") )
            ## average = {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’}

        precision.append( np.array(tmp).mean() )
        precision_err.append( np.array(tmp).std() )

    print(precision)
    print(precision_err)

    plt.errorbar(ks, precision, yerr=precision_err)
    plt.xlabel("k")
    plt.ylabel("Precision")
    plt.title("Precision with average weighting, kNN distance weighting. Q="+ str(q))
    plt.show()


# In[ ]:





# # Linear Regression

# In[54]:


from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(penalty = "none", max_iter=1e3)
#default solver="lbfgs"

LR_model.fit(X, Y)
################################

#from sklearn.preprocessing import PolynomialFeatures
q_poly = [1,2,3,4,5]      


precision=[]
precision_err=[]
for q in q_poly:
    Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X)
    tmp=[]
    #for train, test in kf.split(Xpoly):
    for train, test in kf.split(Xpoly):
        LR_model.fit(Xpoly[train],Y[train])
        ypred = LR_model.predict(Xpoly[test])
        #print(confusion_matrix(ypred,Y[test]))

        tmp.append( precision_score(Y[test], ypred, average="weighted") )
        ## average = {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’}

    precision.append( np.array(tmp).mean() )
    precision_err.append( np.array(tmp).std() )

print(precision)
print(precision_err)

plt.errorbar(q_poly, precision, yerr=precision_err)
plt.xlabel("k")
plt.ylabel("Precision")
plt.title("Precision with LinReg, no penalty")
plt.show()


# In[55]:


from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(penalty = "l2",C=1, max_iter=1e4)

################################

#from sklearn.preprocessing import PolynomialFeatures
q_poly = [1,2,3,4,5]
#Cs = [0.01, 0.1, 1]
Cs = [0.001,0.01,0.1, 1, 10,100,1000]
q=3

precision=[]
precision_err=[]
for c in Cs:
    
    model = LogisticRegression(penalty = "l2",C=c, solver="lbfgs", max_iter=1e5)
    Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X)
    tmp=[]
    #for train, test in kf.split(Xpoly):
    for train, test in kf.split(X):
        model.fit(X[train],Y[train])
        ypred = model.predict(X[test])
        #print(confusion_matrix(ypred,Y[test]))

        tmp.append( precision_score(Y[test], ypred, average="weighted") )
        ## average = {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’}
    #print(tmp)
    precision.append( np.array(tmp).mean() )
    precision_err.append( np.array(tmp).std() )

#print(precision)
#print(precision_err)

plt.errorbar(Cs, precision, yerr=precision_err)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Precision")
plt.title("Precision with LinReg, L2 penalty")
plt.show()


# # SVM Classification

# In[26]:


#precision with averages
def my_precision( ytrue, ypred, classes): #classes = {1,2,...,n}
    if (len(ytrue) != len(ypred) ):
        print("Error. Not same dimensions")
        
    ny = len(ypred) #number of predicitions
    nc = len(classes)
    tp = np.zeros(nc)#true positives
    
    for j in range(ny):
        if (ytrue[j]==ypred[j]):
            tp[ int(ytrue[j]-1) ]+=1
        
    return (sum(tp)/ny) #average precision
    


# In[27]:


from sklearn.svm import LinearSVC
model = LinearSVC(C=c, max_iter=1E4)
classes=np.array([1,2,3,4,5,6,7,8])

################################

#from sklearn.preprocessing import PolynomialFeatures
q_poly = [1,2,3,4,5]
Cs = [0.001,0.01,0.1, 1, 10,100,1000]
#Cs = [0.01,0.1, 1, 10]
q=3

precision=[]
precision_err=[]
for c in Cs:
    model = LinearSVC(C=c, max_iter=1E4)
    #Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X)
    tmp=[]
    #for train, test in kf.split(Xpoly):
    for train, test in kf.split(X):
        model.fit(X[train],Y[train])
        ypred = model.predict(X[test])
        #print(confusion_matrix(ypred,Y[test]))

        #Average weighting on precision
        tmp.append( my_precision(np.array(Y[test]), ypred, classes) )

    precision.append( np.array(tmp).mean() )
    precision_err.append( np.array(tmp).std() )


plt.errorbar(Cs, precision, yerr=precision_err)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Precision")
plt.title("Precision for SVM w/ L2 penalty")
plt.show()


# In[28]:


from sklearn.svm import LinearSVC
#model = LinearSVC(C=c, max_iter=1E3)


###################################


q_poly = [1,2,3,4,5]
Cs = [0.001,0.01,0.1, 1, 10,100,1000]
#Cs = [0.01,0.1, 1, 10]
q=3

precision=[]
precision_err=[]
for c in Cs:
    model = LinearSVC(C=c, max_iter=1E3)
    Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X)
    tmp=[]
    #for train, test in kf.split(Xpoly):
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train],Y[train])
        ypred = model.predict(Xpoly[test])
        #print(confusion_matrix(ypred,Y[test]))

        #Average weighting on precision
        tmp.append( my_precision(np.array(Y[test]), ypred, classes) )

    precision.append( np.array(tmp).mean() )
    precision_err.append( np.array(tmp).std() )


plt.errorbar(Cs, precision, yerr=precision_err)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Precision")
plt.title("Precision for SVM w/ L2 penalty")
plt.show()


# In[31]:


#from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeClassifier

################################

#from sklearn.preprocessing import PolynomialFeatures
q_poly = [1,2,3,4,5]
Cs = [0.001,0.01,0.1, 1, 10,100,1000]
q=3

precision=[]
precision_err=[]
for c in Cs:
    model = RidgeClassifier(alpha = 1/c )
    Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X)
    tmp=[]
    #for train, test in kf.split(Xpoly):
    for train, test in kf.split(X):
        model.fit(X[train],Y[train])
        ypred = model.predict(X[test])
        #print(confusion_matrix(ypred,Y[test]))

        #Average weighting on precision
        tmp.append( my_precision(np.array(Y[test]), ypred, classes) )

    precision.append( np.array(tmp).mean() )
    precision_err.append( np.array(tmp).std() )


plt.errorbar(Cs, precision, yerr=precision_err)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Precision")
plt.title("Precision for Ridge")
plt.show()


# In[ ]:




