#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk

#importing dataset
train = pd.read_csv('train_tweets.csv')
test = pd.read_csv('test_tweets.csv')
train['label'].value_counts()

#cleaning dataset
#cleaning the trainig data
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps= PorterStemmer() 
corpus= []
for i in range(0,31962):
    review = re.sub('[^a-zA-z]',' ',train['tweet'][i])
    review = review.lower()
    review=review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
#cleaning the test dataset
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps= PorterStemmer() 
corpus_test= []
for i in range(0,17197):
    review = re.sub('[^a-zA-z]',' ',test['tweet'][i])
    review = review.lower()
    review=review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_test.append(review)

#making the sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X= cv.fit_transform(corpus).toarray()
y=train.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred)*100)
