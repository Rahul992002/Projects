#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


data = pd.read_csv("C:/Users/nitro/OneDrive/Desktop/loan_approval_dataset.csv")


# In[48]:


data.head()


# In[49]:


data.shape


# In[50]:


data.info()


# In[51]:


data.describe()


# In[52]:


data.columns


# In[53]:


pd.crosstab(data[' cibil_score'] , data[' loan_status'] , margins=True)


# In[61]:


data.boxplot(column=' income_annum')


# In[55]:


sns.histplot(data =data , x=' income_annum' , bins =20 , kde=True )


# In[56]:


data.isnull().sum()


# In[57]:


data['income_annum'] = pd.to_numeric(data[' income_annum'], errors='coerce')


# In[58]:


data[' education'] = pd.to_numeric(data[' education'], errors='coerce')


# In[67]:


data['income'] = np.log(data[' income_annum'])


# In[68]:


sns.histplot(data =data , x='income' , bins =20 , kde=True )


# In[83]:


X = data.drop(columns=[' self_employed']) 
y = data[' self_employed']


# In[84]:


x


# In[85]:


y


# In[88]:


x_train ,x_test , y_train , y_test =train_test_split(x ,y ,test_size =0.2  ,random_state=0)


# In[89]:


print(x_train)


# In[90]:


from sklearn.preprocessing import LabelEncoder


# In[91]:


label_encoder = LabelEncoder()


# In[112]:


data[' education'] = label_encoder.fit_transform(data[' education'])


# In[114]:


data[' loan_status'] = label_encoder.fit_transform(data[' loan_status'])


# In[115]:


x_train


# In[117]:


data[' self_employed'] = label_encoder.fit_transform(data[' self_employed'])


# In[118]:


y_train


# In[119]:


from sklearn.preprocessing import StandardScaler


# In[126]:


data.info()


# In[152]:


ss=StandardScaler()


# In[153]:


# Replace NaN values with 0 in x_train
x_train_filled = np.nan_to_num(x_train, nan=0)

# Ensure y_train is also clean
y_train_filled = np.nan_to_num(y_train, nan=0)

print("x_train after filling NaNs:")
print(x_train_filled)


# In[166]:


x_train_filled = np.nan_to_num(x_train, nan=0, posinf=1e10, neginf=-1e10)
x_test_filled = np.nan_to_num(x_test, nan=0, posinf=1e10, neginf=-1e10)


# In[167]:


x_train_1 = ss.fit_transform(x_train_filled)


# In[168]:


from sklearn.tree import DecisionTreeClassifier


# In[169]:


dtc= DecisionTreeClassifier(criterion='entropy' , random_state=0)


# In[170]:


dtc.fit(x_train_filled,y_train)


# In[171]:


y_pred = dtc.predict(x_test_filled)
y_pred


# In[191]:


y_pred_encoded = label_encoder.fit_transform(y_pred)


# In[200]:


y_pred_encoded


# In[211]:


y_test_encoded = label_encoder.fit_transform(y_test)


# In[212]:


from sklearn import metrics


# In[213]:


auc_score = metrics.accuracy_score(y_pred_encoded,y_test_encoded)


# In[203]:


print(auc_score)


# In[204]:


from sklearn.naive_bayes import GaussianNB


# In[205]:


gb=GaussianNB()
gb.fit(x_train_filled , y_train)


# In[229]:


y_pred_encoded


# In[230]:


auc_score = metrics.accuracy_score(y_pred_encoded,y_test_encoded)
print(auc_score)


# # The above represents that who is eligible for the loan and who is not 
