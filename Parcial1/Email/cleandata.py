# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd  
# Import BeautifulSoup into your workspace - HTML tags
from bs4 import BeautifulSoup  
#Puntuacion y numeros
import re
#Stop words
import nltk
from nltk.corpus import stopwords # Import the stop word list
#Crear caracteristicas
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import csv

import scipy.io as sio

def review_to_words( raw_review ):
	    # Function to convert a raw review to a string of words
	    # The input is a single string (a raw movie review), and 
	    # the output is a single string (a preprocessed movie review)
	    #
	    # 1. Remove HTML
	    
	    review_text = BeautifulSoup(raw_review,"html.parser").get_text() 
	    #
	    # 2. Remove non-letters        
	    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
	    #
	    # 3. Convert to lower case, split into individual words
	    words = letters_only.lower().split()                             
	    #
	    # 4. In Python, searching a set is much faster than searching
	    #   a list, so convert the stop words to a set
	    #nltk.download()
	    stops = set(stopwords.words("english"))                  
	    # 
	    # 5. Remove stop words
	    meaningful_words = [w for w in words if not w in stops]   
	    #
	    # 6. Join the words back into one string separated by space, 
	    # and return the result.
	    return( " ".join( meaningful_words ))   



#train = pd.read_csv('example', header=0,error_bad_lines=False,encoding='iso-8859-14')
train = pd.read_csv("trainingset", header=0, delimiter=";",encoding='iso-8859-14')
#train["email"].fillna(" ", inplace=True) # si hay un email vacio cambia el valor del nan por espacio
#train.dropna(how="all", inplace=True)

#clean_review = review_to_words( train["review"][0] )
#print (clean_review)


# Get the number of reviews based on the dataframe column size
num_reviews = train["email"].size
#print (num_reviews)
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

print ("Cleaning and parsing the training set emails...\n")
# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
	print ("Review %d of %d\n" % ( i+1, num_reviews ))                                             
    clean_train_reviews.append( review_to_words( train["email"][i] ) )
    
    
#print(clean_train_reviews[0])

#Creating Features from a Bag of Words (Using scikit-learn)
print ("Creating the bag of words...\n")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
	                     tokenizer = None,    \
	                     preprocessor = None, \
	                     stop_words = None,   \
	                     max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)
# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

#vocab = vectorizer.get_feature_names()

# Sum up the counts of each vocabulary word
#dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
#for tag, count in zip(vocab, dist):
#    print (count, tag)


np.savetxt("X",train_data_features,fmt='%s', delimiter=" ")
np.savetxt("Y",train["label"],fmt='%s', delimiter=" ")
