#!/usr/bin/env python
# coding: utf-8

# # Version 1.0
# # 4/12/21
# Machine Learning Group Project
# 
# 
# Adapted from file "BasicModelsV1.0.py"

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", size=14) #Font size
plt.rcParams["figure.figsize"] = (10,10) #Fig size


# # Checking csv and titles by importing first row of a file

# In[2]:


file_dir="../SpotifyData/"
file_name="ClassicRock_3Ho3iO0iJykgEQNbjB2sic.csv"

test_csv = pd.read_csv(file_dir+file_name, index_col=0, nrows=1) 
display(test_csv)


# # Compiling Data
# Next step is to make a single dataframe holding the data from all source files
# 

# In[4]:


file_dir="../SpotifyData/"

file_names= ["JazzClassic_37i9dQZF1DXbITWG1ZJKYt.csv",             "CountryHits_1mJhSx6aYQmINsZ8dG4gzU.csv",             "ClassicalStudy_6wObnEPQ63a4kei1sEcMdH.csv",             "HipHopClassics_5CRJExGSQFYXem2hT5GFkd.csv",            "ClassicRock_3Ho3iO0iJykgEQNbjB2sic.csv",             "KpopClassics_0jB4ANR4ox65etDMnxvGLp.csv",             "HeavyMetalClassics_27gN69ebwiJRtXEboL12Ih.csv",             "DanceHits_5oKz4DsTP8zbL97UIPbqp4.csv"]

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
display(data.head())

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


# # LinearClassifier
# 1) Choose q with Linear Classifier
# 
# 
# 2) Move to ridge classifier and choose C

# In[5]:


from sklearn.metrics import precision_score, accuracy_score

from sklearn.model_selection import KFold    
n_splits=5
kf = KFold(n_splits=n_splits, shuffle=True) #important to shuffle for our ordered data


from sklearn.preprocessing import PolynomialFeatures
q_poly = [1,2,3,4,5]      
#Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X)


#Pipelining output from StandardScaler into input of model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler #Rescale data
#pipe = make_pipeline(StandardScaler(), ...model...)


# In[53]:


from sklearn.linear_model import RidgeClassifier
#Imitating Linear Classifier we small alpha
C0 = 1e6

pipe = make_pipeline(StandardScaler(),         RidgeClassifier(alpha = 1/C0 ))

################################

q_poly = [1,2,3,4,5]
accuracy=[]
accuracy_err=[]
for q in q_poly:
    Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X)
    
    tmp=[]
    for train, test in kf.split(Xpoly):
        #Rescale then fit
        pipe.fit(Xpoly[train],Y[train])
        
        #Accuracy
        tmp.append( pipe.score(Xpoly[test], Y[test]) )

    accuracy.append( np.array(tmp).mean() )
    accuracy_err.append( np.array(tmp).std() )


print(accuracy, "\n", accuracy_err)

plt.errorbar(q_poly, accuracy, yerr=accuracy_err, capsize=5)
plt.xticks(q_poly)
plt.xlabel("Highest Order Polynomial Feature (q)")
plt.ylabel("Accuracy")
plt.title("Linear Classifier with Polynomial Features")

plt.savefig("../Images/LinearClassifier_NormalParams.png")    
plt.show()


# In[54]:


#Best accuracy from q_best, save these results for comparison with Ridge Classifier
q_best=3
lin_acc = accuracy[q_best-1]
lin_acc_err=accuracy_err[q_best-1]

print("Best accuracy from Basic Linear Classifier = [%lf +- %lf]" %(lin_acc, lin_acc_err))


# In[55]:


from sklearn.linear_model import RidgeClassifier
#Tuning hyperparameter C now, with q_best

################################

Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X)

#Range of penalty parameters, scaling up by *10
Cs = [0.001,0.01,0.1, 1, 10,100,1000]

accuracy=[]
accuracy_err=[]
for c in Cs:
    pipe = make_pipeline(StandardScaler(),             RidgeClassifier(alpha = 1/c ))
    
    tmp=[]
    for train, test in kf.split(Xpoly):
        pipe.fit(Xpoly[train],Y[train])
        
        #Accuracy
        tmp.append( pipe.score(Xpoly[test], Y[test]) )

    accuracy.append( np.array(tmp).mean() )
    accuracy_err.append( np.array(tmp).std() )

print(accuracy, "\n", accuracy_err)

plt.errorbar(Cs, accuracy, yerr=accuracy_err, capsize=5, label="Ridge Classifier")
plt.hlines( lin_acc + lin_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")
plt.hlines( lin_acc, Cs[0], Cs[-1], linestyles="dashed", color="black", label="Linear Classifier without penalty")
plt.hlines( lin_acc - lin_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")

plt.xticks(Cs)
plt.xscale("log")

plt.xlabel("Penalty Parameter C")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy for Linear-Ridge")
    
#plt.savefig("../Images/RidgeClassifier_NormalParams.png")
plt.show()
#C=1, qbest=3


# # Repeat For Lasso Model based off Linear q_best

# In[6]:


from sklearn.linear_model import LogisticRegression
#Imitating basic model with no penalty

pipe = make_pipeline(StandardScaler(),         LogisticRegression(penalty="none", multi_class="multinomial", solver="sag"))#, tol=1e-3, max_iter=5e2))


#max_iter=1e3
#Doesn't help but kills runtime

#tol=1e-2
#Doesn't help

################################

q_poly = [1,2,3,4,5]
accuracy=[]
accuracy_err=[]
for q in q_poly:
    Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X)
    
    tmp=[]
    for train, test in kf.split(Xpoly):
        #Rescale then fit
        pipe.fit(Xpoly[train],Y[train])
        
        #Accuracy
        tmp.append( pipe.score(Xpoly[test], Y[test]) )

    accuracy.append( np.array(tmp).mean() )
    accuracy_err.append( np.array(tmp).std() )


print(accuracy, "\n", accuracy_err)

plt.errorbar(q_poly, accuracy, yerr=accuracy_err, capsize=5)
plt.xticks(q_poly)
plt.xlabel("Highest Order Polynomial Feature (q)")
plt.ylabel("Accuracy")
plt.title("Logistic Classifier w/o Penalty")

#plt.savefig("../Images/Logistic_AllParams.png")    
plt.show()


# In[7]:


#Best accuracy from q_best, save these results for comparison with Lasso Classifier
q_best=2
log_acc = accuracy[q_best-1]
log_acc_err=accuracy_err[q_best-1]

print("Best accuracy from Basic Logistic Classifier = [%lf +- %lf]" %(log_acc, log_acc_err))


# # Lasso Classifier
# Slight adjustment, using L1 penalty with Logistic rahter than Linear model
# 
# 
# Note: Smaller C encourages sparsity. May be useful to look at which weights are set to 0

# In[8]:


from sklearn.linear_model import LogisticRegression
#Imitating Lasso with Log version


################################

Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X)

#Range of penalty parameters, scaling up by *10
Cs = [0.001,0.01,0.1, 1, 10,100,1000]

accuracy=[]
accuracy_err=[]
for c in Cs:
    pipe = make_pipeline(StandardScaler(),         LogisticRegression(penalty="l1", C=c,                           multi_class="multinomial", solver="saga",                          tol=1e-3, max_iter=5e2
                          ))#, tol=1e-3, max_iter=5e2))

    tmp=[]
    for train, test in kf.split(Xpoly):
        pipe.fit(Xpoly[train],Y[train])
        
        #Accuracy
        tmp.append( pipe.score(Xpoly[test], Y[test]) )

    accuracy.append( np.array(tmp).mean() )
    accuracy_err.append( np.array(tmp).std() )

print(accuracy, "\n", accuracy_err)

plt.errorbar(Cs, accuracy, yerr=accuracy_err, capsize=5, label="Lasso Classifier")
plt.hlines( log_acc + log_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")
plt.hlines( log_acc, Cs[0], Cs[-1], linestyles="dashed", color="black", label="Logistic Classifier without penalty")
plt.hlines( log_acc - log_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")

plt.xticks(Cs)
plt.xscale("log")

plt.xlabel("Penalty Parameter C")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy for Logistic-Lasso")
    
#plt.savefig("../Images/LassoClassifier_AllParams.png")
plt.show()

#C=10, qbest=2


# In[10]:


plt.errorbar(Cs, accuracy, yerr=accuracy_err, capsize=5, label="Lasso Classifier")
plt.hlines( log_acc + log_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")
plt.hlines( log_acc, Cs[0], Cs[-1], linestyles="dashed", color="black", label="Logistic Classifier without penalty")
plt.hlines( log_acc - log_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")

plt.xticks(Cs)
plt.xscale("log")

plt.xlabel("Penalty Parameter C")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy for Logistic-Lasso")
    
#plt.savefig("../Images/LassoClassifier_AllParams.png")
plt.show()


# In[ ]:





# # Support Vector Classifier (SVC)

# In[24]:


from sklearn.svm import LinearSVC

################################

Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X)

#Range of penalty parameters, scaling up by *10
Cs = [0.001,0.01,0.1, 1, 10,100,1000]

accuracy=[]
accuracy_err=[]
for c in Cs:
    pipe = make_pipeline(StandardScaler(),         LinearSVC(C=c, tol=1e-3, max_iter=5e2))

    tmp=[]
    for train, test in kf.split(Xpoly):
        pipe.fit(Xpoly[train],Y[train])
        
        #Accuracy
        tmp.append( pipe.score(Xpoly[test], Y[test]) )

    accuracy.append( np.array(tmp).mean() )
    accuracy_err.append( np.array(tmp).std() )

print(accuracy, "\n", accuracy_err)

plt.errorbar(Cs, accuracy, yerr=accuracy_err, capsize=5, label="Linear SVM Classifier")
plt.hlines( log_acc + log_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")
plt.hlines( log_acc, Cs[0], Cs[-1], linestyles="dashed", color="black", label="Logistic Classifier without penalty")
plt.hlines( log_acc - log_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")

plt.xticks(Cs)
plt.xscale("log")

plt.xlabel("Penalty Parameter C")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy for Linear SVM")
    
#plt.savefig("../Images/SVM_AllParams.png")
plt.show()

#C=1, q=2


# In[25]:


plt.errorbar(Cs, accuracy, yerr=accuracy_err, capsize=5, label="Linear SVM Classifier")
plt.hlines( log_acc + log_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")
plt.hlines( log_acc, Cs[0], Cs[-1], linestyles="dashed", color="black", label="Logistic Classifier without penalty")
plt.hlines( log_acc - log_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")

plt.xticks(Cs)
plt.xscale("log")

plt.xlabel("Penalty Parameter C")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy for Linear SVM")
    
#plt.savefig("../Images/SVM_AllParams.png")
plt.show()


# # Linear SVC for just normalized parameters

# In[36]:


from sklearn.svm import LinearSVC

################################

q_poly = [1,2,3,4,5]

pipe = make_pipeline(StandardScaler(),     LinearSVC(C=1, tol=1e-3, max_iter=5e2))


accuracy=[]
accuracy_err=[]
for q in q_poly:
    Xpoly = PolynomialFeatures(q, include_bias=False).fit_transform(X_norm)

    tmp=[]
    for train, test in kf.split(Xpoly):
        pipe.fit(Xpoly[train],Y[train])
        
        #Accuracy
        tmp.append( pipe.score(Xpoly[test], Y[test]) )

    accuracy.append( np.array(tmp).mean() )
    accuracy_err.append( np.array(tmp).std() )

print(accuracy, "\n", accuracy_err)

plt.errorbar(q_poly, accuracy, yerr=accuracy_err, capsize=5, label="Linear SVM Classifier")
plt.xticks(q_poly)

plt.xlabel("Highest Order Polynomial Feature (q)")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Linear SVM cross validation for q")
    
#plt.savefig("../Images/SVM_NormParams_q.png")
plt.show()


# In[ ]:


#Best accuracy from q_best, save these results for comparison with Lasso Classifier
q_best=2
svm_acc = accuracy[q_best-1]
svm_acc_err=accuracy_err[q_best-1]

print("Best accuracy from Linear SVM Classifier = [%lf +- %lf]" %(svm_acc, svm_acc_err))


# In[38]:


from sklearn.svm import LinearSVC

################################

Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X_norm)

#Range of penalty parameters, scaling up by *10
Cs = [0.001,0.01,0.1, 1, 10,100,1000]

accuracy=[]
accuracy_err=[]
for c in Cs:
    pipe = make_pipeline(StandardScaler(),         LinearSVC(C=c, tol=1e-3, max_iter=5e2))

    tmp=[]
    for train, test in kf.split(Xpoly):
        pipe.fit(Xpoly[train],Y[train])
        
        #Accuracy
        tmp.append( pipe.score(Xpoly[test], Y[test]) )

    accuracy.append( np.array(tmp).mean() )
    accuracy_err.append( np.array(tmp).std() )

print(accuracy, "\n", accuracy_err)

plt.errorbar(Cs, accuracy, yerr=accuracy_err, capsize=5, label="Linear SVM Classifier")
plt.hlines( svm_acc + svm_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")
plt.hlines( svm_acc, Cs[0], Cs[-1], linestyles="dashed", color="black", label="SVM Classifier with C=1")
plt.hlines( svm_acc - svm_acc_err, Cs[0], Cs[-1], linestyles="dashed", color="red")

plt.xticks(Cs)
plt.xscale("log")

plt.xlabel("Penalty Parameter C")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy for Linear SVM")
    
#plt.savefig("../Images/SVM_NormParams.png")
plt.show()

#C=1


# # k Nearest Neighbours

# In[46]:


from sklearn.neighbors import KNeighborsClassifier

ks = np.arange(3,18,2)

accuracy=[]
accuracy_err=[]
for k in ks:
    pipe = make_pipeline(StandardScaler(),         KNeighborsClassifier(n_neighbors=k, weights="distance") )
    
    tmp=[] ## Holds accuracy at each ksplit
    for train, test in kf.split(X):
        pipe.fit(X[train],Y[train])
        #ypred = model.predict(X[test])

        tmp.append( pipe.score(X[test], Y[test]) )

    accuracy.append( np.array(tmp).mean() )
    accuracy_err.append( np.array(tmp).std() )

print(accuracy, "\n", accuracy_err)

plt.errorbar(ks, accuracy, yerr=accuracy_err, capsize=5, label="kNN Classifier")

plt.xticks(ks)

plt.xlabel("Number of neighbours k")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy for kNN")
    
#plt.savefig("../Images/kNN_AllParams.png")
plt.show()
#k=9


# In[47]:


from sklearn.neighbors import KNeighborsClassifier

ks = np.arange(3,18,2)

accuracy=[]
accuracy_err=[]
for k in ks:
    pipe = make_pipeline(StandardScaler(),         KNeighborsClassifier(n_neighbors=k, weights="distance") )
    
    tmp=[] ## Holds accuracy at each ksplit
    for train, test in kf.split(X):
        pipe.fit(X_norm[train],Y[train])
        #ypred = model.predict(X[test])

        tmp.append( pipe.score(X_norm[test], Y[test]) )

    accuracy.append( np.array(tmp).mean() )
    accuracy_err.append( np.array(tmp).std() )

print(accuracy, "\n", accuracy_err)

plt.errorbar(ks, accuracy, yerr=accuracy_err, capsize=5, label="kNN Classifier")

plt.xticks(ks)

plt.xlabel("Number of neighbours k")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy for kNN")
    
#plt.savefig("../Images/kNN_NormParams.png")
plt.show()
#k=5


# # Comparing across parameters

# In[51]:


from sklearn.svm import LinearSVC

################################
q_best=2; c=1
Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X_norm)
pipe = make_pipeline(StandardScaler(),     LinearSVC(C=c, tol=1e-3, max_iter=1e3))

tmp=[]
for train, test in kf.split(Xpoly):
    pipe.fit(Xpoly[train],Y[train])

    #Accuracy
    tmp.append( pipe.score(Xpoly[test], Y[test]) )

accuracy_svm_norm=[0,0]
accuracy_svm_norm[0]= np.array(tmp).mean()
accuracy_svm_norm[1]= np.array(tmp).std()
print(accuracy_svm_norm)


# In[52]:


from sklearn.svm import LinearSVC

################################
q_best=2; c=1
Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X)
pipe = make_pipeline(StandardScaler(),     LinearSVC(C=c, tol=1e-3, max_iter=1e3))

tmp=[]
for train, test in kf.split(Xpoly):
    pipe.fit(Xpoly[train],Y[train])

    #Accuracy
    tmp.append( pipe.score(Xpoly[test], Y[test]) )

accuracy_svm_all=[0,0]
accuracy_svm_all[0]= np.array(tmp).mean()
accuracy_svm_all[1]= np.array(tmp).std()
print(accuracy_svm_all)



#C=1, q=2


# In[54]:


from sklearn.linear_model import LogisticRegression
#Imitating Lasso with Log version

################################
q_best=2; c=10
Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X)

pipe = make_pipeline(StandardScaler(),         LogisticRegression(penalty="l1", C=c,                           multi_class="multinomial", solver="saga",
                           max_iter=1e3))

tmp=[]
for train, test in kf.split(Xpoly):
    pipe.fit(Xpoly[train],Y[train])

    #Accuracy
    tmp.append( pipe.score(Xpoly[test], Y[test]) )

accuracy_lasso_all=[0,0]
accuracy_lasso_all[0]= np.array(tmp).mean()
accuracy_lasso_all[1]= np.array(tmp).std()
print(accuracy_lasso_all)

#C=10, qbest=2


# In[57]:


from sklearn.linear_model import LogisticRegression
#Imitating Lasso with Log version

################################
q_best=2; c=10
Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X_norm)

pipe = make_pipeline(StandardScaler(),         LogisticRegression(penalty="l1", C=c,                           multi_class="multinomial", solver="saga",
                           max_iter=1e3))

tmp=[]
for train, test in kf.split(Xpoly):
    pipe.fit(Xpoly[train],Y[train])

    #Accuracy
    tmp.append( pipe.score(Xpoly[test], Y[test]) )

accuracy_lasso_norm=[0,0]
accuracy_lasso_norm[0]= np.array(tmp).mean()
accuracy_lasso_norm[1]= np.array(tmp).std()
print(accuracy_lasso_norm)

#C=10, qbest=2


# In[55]:


from sklearn.linear_model import RidgeClassifier

################################
q_best=3; c=10
Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X)
pipe = make_pipeline(StandardScaler(),         RidgeClassifier(alpha = 1/c ))

tmp=[]
for train, test in kf.split(Xpoly):
    pipe.fit(Xpoly[train],Y[train])
    #Accuracy
    tmp.append( pipe.score(Xpoly[test], Y[test]) )

accuracy_ridge_all=[0,0]
accuracy_ridge_all[0]= np.array(tmp).mean()
accuracy_ridge_all[1]= np.array(tmp).std()
print(accuracy_ridge_all)
#C=1, qbest=3

#Instant result!


# In[56]:


from sklearn.linear_model import RidgeClassifier

################################
q_best=3; c=10
Xpoly = PolynomialFeatures(q_best, include_bias=False).fit_transform(X_norm)
pipe = make_pipeline(StandardScaler(),         RidgeClassifier(alpha = 1/c ))

tmp=[]
for train, test in kf.split(Xpoly):
    pipe.fit(Xpoly[train],Y[train])
    #Accuracy
    tmp.append( pipe.score(Xpoly[test], Y[test]) )

accuracy_ridge_norm=[0,0]
accuracy_ridge_norm[0]= np.array(tmp).mean()
accuracy_ridge_norm[1]= np.array(tmp).std()
print(accuracy_ridge_norm)
#C=1, qbest=3

#Instant result!


# In[58]:


from sklearn.neighbors import KNeighborsClassifier

k=9
pipe = make_pipeline(StandardScaler(),     KNeighborsClassifier(n_neighbors=k, weights="distance") )
tmp=[]
for train, test in kf.split(X):
    pipe.fit(X[train],Y[train])
    #Accuracy
    tmp.append( pipe.score(X[test], Y[test]) )

accuracy_kNN_all=[0,0]
accuracy_kNN_all[0]= np.array(tmp).mean()
accuracy_kNN_all[1]= np.array(tmp).std()
print(accuracy_ridge_all)
#k=9


# In[59]:


from sklearn.neighbors import KNeighborsClassifier

k=5
pipe = make_pipeline(StandardScaler(),     KNeighborsClassifier(n_neighbors=k, weights="distance") )
tmp=[]
for train, test in kf.split(X_norm):
    pipe.fit(X_norm[train],Y[train])
    #Accuracy
    tmp.append( pipe.score(X_norm[test], Y[test]) )

accuracy_kNN_norm=[0,0]
accuracy_kNN_norm[0]= np.array(tmp).mean()
accuracy_kNN_norm[1]= np.array(tmp).std()
print(accuracy_kNN_norm)
#k=5


# # Combining best results from each model

# In[64]:


print(accuracy_ridge_norm,"\n",      accuracy_lasso_norm,"\n",       accuracy_svm_norm,"\n",      accuracy_kNN_norm) 

print("\n\n")
print(accuracy_ridge_all,"\n",      accuracy_lasso_all,"\n",       accuracy_svm_all,"\n",      accuracy_kNN_all)


#Ridge very fast!


# In[ ]:




