#The goal of both stemming and lemmatization is to condense derived words into their base forms

#  Stemming is faster as it chops off the end of the word using heuristics without any understanding of the context which the word is used.

# Lemmatization uses more informed analysis to create groups of words with similar meaning based on the context around the word.It will always return a dictionary word
# Lemmatization maps each word to a word that is in its dictionary and has a meaningful context.

# Stemming -> Faster but less accurate
# Lemmatization -> Slower but accurate

# Stemming
import nltk
ps = nltk.PorterStemmer()
# Try a simple example
# print(ps.stem('grows'))
# print(ps.stem('growing'))
# print(ps.stem('grew'))


import pandas as pd
import string
import re

pd.set_option('display.max_colwidth', 100)
data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
data.columns = ['label', 'text']

stopwords = nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation]) # remove punctuation
    tokens = re.split('\W+',text)# tokenization
    text = [word for word in tokens if word not in stopwords]
    return text

data['text'] = data['text'].apply(lambda x: clean_text(x))

#print(data.head())


# NOW THE STEMMING
def stemming(text):
    return [ps.stem(word) for word in text]

data['text'] = data['text'].apply(lambda x: stemming(x))

#print(data.head())




# Lemmatizing

ws = nltk.WordNetLemmatizer()

# A case where porter stemmer fails and how lemmatizer does the job correctly

print ("Porter Stemmer output of the word goose is",ps.stem('goose'))
print("Porter Stemmer output of the word goose is",ps.stem('geese'))
print("Lemmatizer output of the word goose is " , ws.lemmatize('goose'))
print("Lemmatizer output of the word geese is ", ws.lemmatize('geese'))
# As we notice, the lemmatizer does a better job than the porter stemmer

def lemmatizer(text):
    return [ws.lemmatize(word) for word in text]

data['text'] = data['text'].apply(lambda x: lemmatizer(x))

print(data.head())





