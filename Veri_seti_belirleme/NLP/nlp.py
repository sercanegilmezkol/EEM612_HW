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
        return 1  '''
    
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