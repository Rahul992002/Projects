#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt   


# In[2]:


data = pd.read_csv("C:/Users/nitro/OneDrive/Desktop/covid19_italy_region.csv")


# In[3]:


print(data)


# In[4]:


data.columns


# In[7]:


data.describe()


# In[8]:


sns.relplot(x='TotalPositiveCases' , y='Deaths' ,hue='Recovered' , data =data)


# In[12]:


sns.pairplot(data)


# In[13]:


data.columns


# In[16]:


sns.relplot(x='TotalPositiveCases' , y='Recovered' , kind = 'line' , data=data)


# In[19]:


data.columns


# In[20]:


data.head(5)


# In[21]:


data.tail(5)


# In[ ]:




