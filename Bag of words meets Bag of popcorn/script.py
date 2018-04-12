# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:58:53 2018

@author: Shikhar
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from  bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
train_label=pd.read_csv("labeledTrainData.tsv",delimiter="\t",quoting=3)
train_unlabel=pd.read_csv("unlabeledTrainData.tsv",delimiter="\t",quoting=3)
test=pd.read_csv("testData.tsv",delimiter="\t",quoting=3)

corpus=[]
corpus2=[]
for i in range(0,25000):
    review=re.sub('[^A-Za-z]',' ',train_label['review'][i])
    review=BeautifulSoup(review).get_text()
    review=review.lower()
    review=review.split()
    review3=re.sub('[^A-Za-z]',' ',test['review'][i])
    review3=BeautifulSoup(review3).get_text()
    review3=review3.lower()
    review3=review3.split()
    ps=PorterStemmer()
    
    review=[ps.stem(i) for i in review if not i in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
    review3=[ps.stem(i) for i in review3 if not i in set(stopwords.words('english'))]
    review3=' '.join(review3)
    corpus2.append(review3)
   




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 6000)
x = cv.fit_transform(corpus).toarray()
x1=cv.fit_transform(corpus2).toarray()
y = train_label.iloc[:, 1].values

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x, y)

# Predicting the Test set results
y_pred = classifier.predict(x1)

submissionhas=pd.DataFrame({'id':test['id'].values,'sentiment':y_pred
                        })
submissionhas.to_csv('bag4.csv', index=False)
