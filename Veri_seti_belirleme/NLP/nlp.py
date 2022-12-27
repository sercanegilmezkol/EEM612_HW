# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:50:23 2022

@author: sercan.egilmezkol
"""

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re
import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv')

yorumlar.Liked.value_counts().plot(kind="bar")
yorumlar.Liked.value_counts()


ps = PorterStemmer()

nltk.download('stopwords')


def olumsuzluk_Tespit(yorum):
    # Olumsuzluk_Dataset=["not","hasn't","didn't"]
    # ni = yorum boyu

    _Not = 0
    ReturnSend = 0
    yrm = ""

    for i in range(len(yorum)):
        if _Not == 0 and yorum[i] == "n":
            _Not = 1
            yrm += yorum[i]
        elif _Not == 1 and (yorum[i] == "o" or yorum[i] == "'"):
            _Not = 2
            yrm += yorum[i]
        elif _Not == 2 and yorum[i] == "t":
            _Not = 3
            yrm += yorum[i]
        elif _Not == 3 and yorum[i] == " ":
            _Not = 4
            ReturnSend = 1
            yrm += "_"
        else:
            _Not = 0
            yrm += yorum[i]
    '''if ReturnSend == 0:
        return 0
    else:
        return 1   '''
    
    return yrm

def olumsuzluk_Tespit2(yorum):
    # Olumsuzluk_Dataset=["not","hasn't","didn't", ...]
    _Not = 0
    yrm = ""

    for i in range(len(yorum)):
        if _Not == 0 and yorum[i] == "n":
            _Not = 1
            yrm += yorum[i]
        elif _Not == 1 and (yorum[i] == "o" or yorum[i] == "'"):
            _Not = 2
            yrm += yorum[i]
        elif _Not == 2 and yorum[i] == "t":
            _Not = 3
            yrm += yorum[i]
        elif _Not == 3 and yorum[i] == " ":
            _Not = 4
            yrm += " EEM612_SE "
        else:
            _Not = 0
            yrm += yorum[i]       
    
    return yrm


#Preprocessing (Önişleme)
derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]', ' ', yorumlar['Review'][i])
    yorum = yorum.lower()
    gecici_yrm = yorumlar.iloc[i, 0]
    '''if olumsuzluk_Tespit(gecici_yrm):
        if yorumlar.iloc[i, 1] == 0:
            yorumlar.iloc[i, 1] = 1   '''
        #else:
            #yorumlar.iloc[i,1] = 0
    #yorum = olumsuzluk_Tespit(gecici_yrm)
    '''if yorumlar.iloc[i, 1] == 0:
        yorum = olumsuzluk_Tespit2(gecici_yrm)   '''
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(
        stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)


# Feautre Extraction ( Öznitelik Çıkarımı)
# Bag of Words (BOW)
gosterge=600
print(gosterge)
cv = CountVectorizer(max_features=gosterge)
x = cv.fit_transform(derlem).toarray()  # bağımsız değişken
y = yorumlar.iloc[:, 1].values  # bağımlı değişken
#print(cv.get_feature_names_out())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0)

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
cm_display = ConfusionMatrixDisplay(cm).plot()








'''
from nltk import SyllableTokenizer

string_for_stemming = """The crew of the USS Discovery discovered many discoveries.
Discovering is what explorers do."""

tk = SyllableTokenizer()
        
geek = tk.tokenize(string_for_stemming)

print(geek)
'''


'''

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
'''


''' Almış olduğum bazı notlar
def fac(n):
    f=1
    for i in range (2,n+1):
        f *= i
    return f
print(fac(5))

def goster(n):
    f=10
    if n <0  :
        print('Negative changed to zero')
    elif n == 0:
        print('Zero state')
    else:
        print('Entered to for loop')
    for i in range (1,n+1):
        f = i
    return f
print(goster(2))
'''
