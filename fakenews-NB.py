#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import joblib


# In[27]:


df = pd.read_csv("news.csv")


# In[28]:


df.head()


# In[29]:


df.isna().sum()


# In[30]:


df['textlen'] = df['text'].apply(len)


# In[31]:


df.head()


# In[32]:


df["label"].value_counts()


# In[33]:


sns.countplot(x="label", data=df)


# In[34]:


df.head()


# In[35]:


df['text'][0].lower()


# In[36]:


df['text']=df['text'].str.lower()


# In[37]:


df.head()


# In[38]:


X=df['text']
y=df['label']


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)


# In[40]:


y_train.value_counts()


# In[41]:


y_test.value_counts()


# In[42]:



  

# Create a Vectorizer Object
vectorizer = CountVectorizer(stop_words='english')
  
vectorizer.fit(X_train)


# In[43]:



vector_train = vectorizer.transform(X_train)
  
# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector_train.toarray())


# In[44]:


vector_test = vectorizer.transform(X_test)


# In[45]:


clf = MultinomialNB()
clf.fit(vector_train, y_train)


# In[46]:


y_pred = clf.predict(vector_test)
m = y_test.shape[0]
n = (y_test != y_pred).sum()
print("SVM Accuracy = " + format((m-n)/m*100, '.2f') + "%")


# In[ ]:





# In[47]:




#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

print(cf_matrix)


# In[48]:


plot_confusion_matrix(clf,vector_test,y_test)


# In[49]:


print(classification_report(y_test,y_pred))
    


# In[ ]:





# In[ ]:





# In[ ]:





# # method 2

# In[51]:



# Define the Pipeline
"""
Step1: get the oultet binary columns
Step2: pre processing
Step3: Train a Random Forest Model
"""

model_pipeline = Pipeline(steps=[('vectorizer',CountVectorizer(stop_words='english')),('clf',MultinomialNB())])
        


# In[52]:


# fit the pipeline with the training data
#xtrain, ytrain
model_pipeline.fit(X_train,y_train)



# predict target values on the training data
model_pipeline.predict(X_test)


# In[53]:


y_pred = model_pipeline.predict(X_test)
m = y_test.shape[0]
n = (y_test != y_pred).sum()
print("SVM Accuracy = " + format((m-n)/m*100, '.2f') + "%")


# In[54]:


#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

print(cf_matrix)


# In[55]:


plot_confusion_matrix(clf,vector_test,y_test)


# In[ ]:





# In[ ]:




