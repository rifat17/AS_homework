#!/usr/bin/env python
# coding: utf-8

# In[2]:


# IMPORTS HERE

import numpy as np

# import matplotlib
from matplotlib import pyplot

from scipy.stats import pearsonr


# In[3]:


ageList = [23,23,27,27,39,41,47,49,50,52,54,54,56]
fatList = [9.5,26.5,7.8,17.8,31.4,25.9,27.4,27.2,31.2,34.6,42.5,28.8,33.4]


# In[4]:


ageList


# In[5]:


fatList


# In[6]:


len(ageList)


# In[7]:


len(fatList)


# In[8]:


type(ageList)


# In[9]:


npAge = np.array(ageList)


# In[10]:


npAge


# In[11]:


type(npAge)


# In[12]:


npAge.ndim


# In[13]:


npFat = np.array(fatList)


# In[14]:


npFat


# In[20]:


ageBar = npAge.sum()/len(npAge)


# In[19]:


np.sum(ageList)/len(ageList)


# In[21]:


fatBar = npFat.sum()/len(npFat)


# In[22]:


fatBar


# In[24]:


ageDif = npAge - ageBar


# In[25]:


ageDif


# In[26]:


fatDif = npFat - fatBar


# In[27]:


fatDif


# In[28]:


fatDifAgeDif = fatDif * ageDif


# In[29]:


fatDifAgeDif


# In[30]:


sumfatDifAgeDif = fatDifAgeDif.sum()


# In[31]:


sumfatDifAgeDif


# In[32]:


delA = np.sqrt(np.sum( np.square( ageDif ) ) / len(ageList))


# In[33]:


delA


# In[34]:


delB = np.sqrt(np.sum( np.square( fatDif ) ) / len(fatList))


# In[35]:


delB


# In[38]:


r = sumfatDifAgeDif / (len(ageList) * delA * delB)


# In[39]:


r


# In[42]:


scipyr,_ = pearsonr(ageList,fatList)


# In[43]:


scipyr


# In[46]:


diff = abs(r - scipyr)


# In[47]:


diff


# In[49]:


pyplot.scatter(ageList,fatList)


# In[ ]:





# In[ ]:




