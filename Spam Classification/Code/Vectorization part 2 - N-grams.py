#!/usr/bin/env python
# coding: utf-8

# # N- Grams

# It creates a document-term matrix where instead of the columns representing single words, they represent all combinations of adjacent words of lenght 'n' in your text

# In[9]:


import pandas as pd
import string
import re
import nltk
ps = nltk.PorterStemmer()
pd.set_option('display.max_colwidth', 100)
data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
data.columns = ['label', 'text']

stopwords = nltk.corpus.stopwords.words('english')


# In[12]:


# Remove punctuation , tokenize and remove stopwwords

def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation]) # remove punctuation
    tokens = re.split('\W+',text)# tokenization
    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return text

data['text'] = data['text'].apply(lambda x: clean_text(x))
data.head(5)


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
ngram_vectorize = CountVectorizer(ngram_range = (2,2)) # only look for bi-grams
x_counts =  ngram_vectorize.fit_transform(data['text'])
print (x_counts.shape)
print (ngram_vectorize.get_feature_names())


# # Apply N-Grams to a smaller dataset for inspection

# In[23]:


data_sub = data[0:10]
from sklearn.feature_extraction.text import CountVectorizer
ngram_vectorize = CountVectorizer(ngram_range = (2,2)) # only look for bi-grams
x_counts =  ngram_vectorize.fit_transform(data_sub['text'])
print (x_counts.shape)
print (ngram_vectorize.get_feature_names())


# In[28]:


df = pd.DataFrame(x_counts.toarray())
df.columns = ngram_vectorize.get_feature_names()
df.head(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




