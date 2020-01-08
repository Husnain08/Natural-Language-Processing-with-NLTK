# Feature creation

# There are 2 hypothesis that we'll work with. One is that spam messages tend to be longer than non-spam messages and the
# second hypothesis is that the spam messages contains more punctuation than non-spam messages
import matplotlib
import pandas as pd
import string
import re

data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
data.columns = ['label', 'text']
# Creating a feature for text message length
data['text_len'] = data['text'].apply(lambda x: len(x) - x.count(" ")) # don't count white space

# Create a feature for the percentage of punctuation in the text message
punctuation = string.punctuation
def count_punc(text):
    count = sum([1 for char in text if char in punctuation])
    return count/round(len(text) - text.count(" "),2)*100

data['percentage_punct'] = data['text'].apply(lambda x: count_punc(x))
print(data.head(5))


# Evaluate the features that we created

# The text lenght feature
import matplotlib.pyplot as plt
import numpy as np
bins = np.linspace(0,200,40)
plt.hist(data[data['label'] == 'spam']['text_len'] , normed=True, bins=bins , label='spam')
plt.title("Histogram of the lenght of Spam Emails")
plt.hist(data[data['label'] == 'ham']['text_len'],normed=True,bins=bins , label='ham')
plt.legend(loc = 'upper left')
plt.title("Histogram of the lenght of Non-spam Emails")
plt.show()


# Punctuation percentage feature

bins = np.linspace(0,50,40)
plt.hist(data[data['label'] == 'spam']['percentage_punct'] , normed=True, bins=bins , label='spam')
plt.title("Histogram of the punctuation percentage of Spam Emails")
plt.hist(data[data['label'] == 'ham']['percentage_punct'],normed=True,bins=bins, label='ham')
plt.legend(loc = 'upper left')
plt.title("Histogram of the punctuation percentage of Non-spam Emails")
plt.show()


# Box - Cox Power Transformation

## Transformation process
# 1. Choose what range of exponents to test
# 2. Apply each transformation to each value of the chosen feature ( usually a skewed feature and having outliers)
# 3. Use some criteria to determine which of the transformations yielded the best distributions

# We will use that transformation which makes the data roughly like normal distribution

for i in [1 ,2,3,4,5]:
    plt.hist((data['percentage_punct'])**(1/i) , bins = 40)
    plt.title("Transformation: 1/{}".format(str(i)))
    plt.show()






