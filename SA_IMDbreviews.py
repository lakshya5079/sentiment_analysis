#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk

# reading the dataset
reviews_train = []
reviews_test = []
for line in open('full_train.txt','r'):
    reviews_train.append(line.strip())

for line in open('full_test.txt','r'):
    reviews_test.append(line.strip())

#cleaning dataset
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

#making sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(reviews_train_clean)
X_test = cv.fit_transform(reviews_test_clean)

#fitting logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

    
lr = LogisticRegression(C=0.05)
lr.fit(X_train, y_train)
print( accuracy_score(y_val, lr.predict(X_val))*100)
    
#     Accuracy for C=0.01: 0.87872
#     Accuracy for C=0.05: 0.88704
#     Accuracy for C=0.25: 0.88224
#     Accuracy for C=0.5: 0.88
#     Accuracy for C=1: 0.8776

# Final Accuracy: 0.88704
    
