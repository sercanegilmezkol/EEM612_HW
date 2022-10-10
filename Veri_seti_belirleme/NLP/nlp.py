# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:50:23 2022

@author: sercan.egilmezkol
"""

import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv')

yorumlar.Liked.value_counts().plot(kind="bar")
yorumlar.Liked.value_counts()


import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing (Önişleme)
derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)

   
#Feautre Extraction ( Öznitelik Çıkarımı)
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = yorumlar.iloc[:,1].values # bağımlı değişken

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

print(yorumlar)


from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


y = yorumlar.Liked
x = yorumlar.Review

cv = CountVectorizer()
X = cv.fit_transform(x)
cv.get_feature_names()[:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import RandomizedSearchCV
parameter_space = {
    'hidden_layer_sizes': [(1024), (50,), (50,100, 50), (48,), (48, 48, 48), (96,), (144,), (192,), (96, 144, 192), (240,), (144, 192, 240)],
    'activation': ['tanh', 'logistic', 'relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.05, 0.1, 1],
    'beta_1': [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
    'beta_2': [0.990, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999],
    'learning_rate': ['constant','adaptive'],
                }

mlp = MLPClassifier(max_iter=200, random_state=42)

score = ['accuracy', 'precision']
clf = RandomizedSearchCV(mlp, parameter_space, n_jobs = -1, n_iter = 15,  cv=3, refit='precision', scoring=score, random_state=0)

from sklearn.metrics import accuracy_score
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print("Validation Accuracy",score*100,"%")
print(yorumlar)
