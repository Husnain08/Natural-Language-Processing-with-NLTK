# STEP -1 (Removing Punctuation)
# Although we can use the nltk toolbox for removing punctuation, it is also possible (and easy) to define our own function for this task

import pandas as pd
import string
import re

pd.set_option('display.max_colwidth', 100)
data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
data.columns = ['label', 'text']

punctuation = string.punctuation


def remove_punctuation(text):
    no_punctuation = "".join([char for char in text if char not in string.punctuation])
    return no_punctuation

#
# data['text'] = ((data['text']).apply(lambda x: remove_punctuation(x)))
#
# print(data.head())


# STEP -2 (TOKENIZATION)

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


#
data['clean_text'] = data['text'].apply(lambda x: tokenize(x.lower()))

#print (data.head(5))


# PART -3 (REMOVE STOPWORDS)
import nltk
stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    text_ = "".join([word for word in text if word not in stopwords])
    return text_

data['clean_text'] = data['text'].apply(lambda x: remove_stopwords(x.lower()))
print (data.head(5))