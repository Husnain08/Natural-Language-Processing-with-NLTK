# PART 1 - Reading Semi-structured data ( Data preparation )

# read the data file
import pandas as pd
raw_data = open('SMSSpamCollection.tsv' ).read()
# print the data
#print (raw_data[0:500])

parsed_data = raw_data.replace('\t' , '\n').split('\n')
#print(parsed_data[0:5])
# we note that the labels are at every other position, so we can pull out those labels and create a second list. We can do this
#for extracting the text, too

labels = parsed_data[0:len(parsed_data):2]
#print (labels)

textlist = parsed_data[1::2]
#print(textlist[0:2])

dataframe = pd.DataFrame({
'label' : labels[:-1],
    'text': textlist
})

#print(dataframe.head(5))

# PART - 2 EXPLORING THE DATASET

data = pd.read_csv('SMSSpamCollection.tsv' , sep ='\t' , header = None) # there are no labels in this dataframe yet
data.columns = ['labels' , 'text']
# shape of data
#print ("The data has {} rows and {} columns".format(data.shape[0] , data.shape[1]))

# Label distribution to check if the data is imbalanced

#print("Out of {} labels, {} are spam and {} are ham".format(data.shape[0] , len(data[data['labels'] == 'spam']) , len(data[data['labels'] == 'ham'])))

# Checking null values

#print("Total null values in the labels  are {}".format(data['labels'].isnull().sum()))
#print("Total null values in the text  are {}".format(data['text'].isnull().sum()))


# PART -3 Regular expressions
import re
sentence_1 = 'This is a clean sentence which has no special characters or has difficult patterns'
difficult_sentence_1 = 'This     sentence     has     large     spaces      between      the      words'
difficult_sentence_2 = 'This---sentence***has%%%many&&&&special&&&&characters@@@@between****the......original*&^text'

# splitting a sentence into a list of words

# using the '\s' identifier for cleaning the first sentence

sentence_1_cleaned = re.split('\s' , sentence_1) # It looks for a single white space to split the sentences
#print(sentence_1_cleaned)

# To look for multiple white spaces
difficult_sentence_1_cleaned = re.split('\s+' , difficult_sentence_1) # It looks for a single white space to split the sentences
#print(difficult_sentence_1_cleaned)


difficult_sentence_2_cleaned = re.split('\W+' , difficult_sentence_2)  # 'W+ looks for any non-word character to define the split'

#print(difficult_sentence_2_cleaned)

# We can also use findall function of regex to ignore the characters in our text instead of searching the whole text file.

difficult_sentence_1_cleaned = re.findall('\S+' , difficult_sentence_1)  # It looks for non white charaters and return them
print(difficult_sentence_1_cleaned)


difficult_sentence_2_cleaned = re.findall('\w+' , difficult_sentence_2) # It looks for a single word characters and returns them
print(difficult_sentence_2_cleaned)




