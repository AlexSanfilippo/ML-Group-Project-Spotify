#!/usr/bin/env python
# coding: utf-8

# # 25/11/21
# Machine Learning Group Project
# 
# 
# This file is our main file for the project.  All of our code will be contained
# here.
# 
# 
# Initial file creation by Dylan Kierans

# # Version 0.1

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", size=14)


# # Checking csv and titles by importing first row of a file

# In[52]:


file_dir="../SpotifyData/"
file_name="ClassicRock_3Ho3iO0iJykgEQNbjB2sic.csv"

test_csv = pd.read_csv(file_dir+file_name, index_col=0, nrows=1) 
display(test_csv)


# # Choosing input columns
# Now just importing the necessary training parameters

# In[55]:


#Sticking with just the basic parameters for now
parameter_columns = [1,2,4,6,7,8,9]

#Saving these other ones for later
# key_column=3
# mode_column=5
# tempo_column=11
# duration_column=17


#Testing to make sure read column numbers correctly
data = pd.read_csv(file_dir+file_name, usecols=parameter_columns) 
display(data.head())


# # Playlist_number Column
# Now to add an extra column which will hold the playlist_number

# In[ ]:


# Can never be too safe with printing len() values
print(len(data))

#Want to add column with name "playlist_number" and all entries equal to playlist number
playlist_number=1
playlist_numbers = playlist_number*np.ones(len(data))

data["playlist_number"]=playlist_numbers
display(data)
#Works a charm


# # Compiling Data
# Next step is to make a single dataframe holding the data from all source files

# In[56]:


file_names= ["ClassicRock_3Ho3iO0iJykgEQNbjB2sic.csv",             "ClassicalStudy_6wObnEPQ63a4kei1sEcMdH.csv",             "CountryHits_1mJhSx6aYQmINsZ8dG4gzU.csv",             "DanceHits_5oKz4DsTP8zbL97UIPbqp4.csv",             "HeavyMetalClassics_27gN69ebwiJRtXEboL12Ih.csv",             "HipHopClassics_5CRJExGSQFYXem2hT5GFkd.csv",             "JazzClassic_37i9dQZF1DXbITWG1ZJKYt.csv",             "KpopClassics_0jB4ANR4ox65etDMnxvGLp.csv"]


# Most convenient to set up dataframe with initial file, then systematically loop through the others
#Sticking with just the basic parameters for now
parameter_columns = [1,2,4,6,7,8,9]
nrows=200



#1 indexing here more natural
playlist_number=1
data = pd.read_csv(file_dir+file_names[playlist_number-1], usecols=parameter_columns, nrows=nrows)

playlist_numbers = playlist_number*np.ones(len(data))
data["playlist_number"]=playlist_numbers #Now has extra playlist column



# Now looping through the remaining filenames
for playlist_number in range(2,len(file_names)+1):
    # dtmp := Temporary dataframe, will append to df.data once data filtered and prepped
    dtmp= pd.read_csv(file_dir+file_names[playlist_number-1], usecols=parameter_columns, nrows=nrows)

    playlist_numbers = playlist_number*np.ones(len(dtmp)) # array of playlist_number
    dtmp["playlist_number"]=playlist_numbers #Now has extra playlist column
    data=data.append(dtmp, ignore_index=True) #Append back to main df.data
    
display(data)


# # Training?
# Any pre-processing necessary for:
# - Repeat songs
# - Incomplete data
# - ...

# In[57]:


# NOTE: LOUDNESS IS NOT NORMALIZED OOPS


# In[ ]:




