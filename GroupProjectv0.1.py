###26th October, 2021
#Machine Learning Group Project
#
#This file is our main file for the project.  All of our code will be contained
#here.  To modify, first make a branch of main, then make your changes, then
#request a merge with a description of your changes.
#initial file creation by Alex Sanfilippo


#Some basic libraries we will likely need
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot  as plt #for data visualization

df = pd.read_csv("OURDATA.csv") #read in the spotify data into a dataframe

#Feature and Output seperation
X1=df.iloc[:,0] #assign first input feature to X1
X2=df.iloc[:,1] #assign second input feature to X2
#X1, X2, ... XN for N features
Xwhole=np.column_stack((X1,X2)) #put all inputs into one matrix
ywhole=df.iloc[:,2]  #output data
print('Hello world')
