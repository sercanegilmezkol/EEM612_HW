# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:08:22 2022

@author: sercan.egilmezkol
"""


#Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Veri yukleme
veriler1 = pd.read_csv('athlete_events.csv')
#print(veriler1)

veriler2 = veriler1.dropna(subset=['Height','Weight'])
#print(veriler2)

veriler3 = veriler2.drop_duplicates(subset="Name", keep='first')
#print(veriler3)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
cinsiyet = veriler3.iloc[:,2:3].values
cinsiyet[:,0] = le.fit_transform(veriler3.iloc[:,2:3])
ohe = preprocessing.OneHotEncoder()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
#print(cinsiyet)
cinsiyet = pd.DataFrame(data=cinsiyet[:,0:1], index = range(3132), columns = ['Sex'])
#cinsiyet = pd.DataFrame(data=cinsiyet[:,0:1], index = range(99041), columns = ['Sex'])
#print(cinsiyet)



le = preprocessing.LabelEncoder()
brans = veriler3.iloc[:,13:14].values
brans[:,0] = le.fit_transform(veriler3.iloc[:,13:14])
ohe = preprocessing.OneHotEncoder()
brans = ohe.fit_transform(brans).toarray()
#print(brans)
brans = pd.DataFrame(data=brans, index = range(3132))
#brans2 = pd.DataFrame(data=brans, index = range(99041))
#print(brans)



sutun_01 = veriler3[['Height','Weight']]
sutun_01 = sutun_01.iloc[:,:].values
sutun_01 = pd.DataFrame(data=sutun_01, index = range(3132), columns = ['Height','Weight'])
#sutun_01 = pd.DataFrame(data=sutun_01, index = range(99041), columns = ['Height','Weight'])
#print(sutun_01)

veriler_son = pd.concat([cinsiyet,sutun_01], axis=1)
print(veriler_son)


"""
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
"""
"""
Ploting = veriler_son.iloc[0:439,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
plt.plot(Ploting.iloc[0:439,1:2].values,Ploting.iloc[0:439,2].values,'r.')

Ploting = veriler_son.iloc[439:844,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
plt.plot(Ploting.iloc[0:405,1:2].values,Ploting.iloc[0:405,2].values,'r.')

Ploting = veriler_son.iloc[844:878,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
plt.plot(Ploting.iloc[0:34,1:2].values,Ploting.iloc[0:34,2].values,'g.')

Ploting = veriler_son.iloc[878:2013,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
plt.plot(Ploting.iloc[0:1135,1:2].values,Ploting.iloc[0:1135,2].values,'b.')

Ploting = veriler_son.iloc[2013:2531,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
plt.plot(Ploting.iloc[0:518,1:2].values,Ploting.iloc[0:518,2].values,'y.')
"""


"""
    Archery Men's Individual
    Archery Women's Individual
    Athletics Men's 10 kilometres Walk
    Athletics Men's 100 metres
    Athletics Men's High Jump
    Athletics Men's 10,000 metres

.... için aşağıdaki şekilde sıralanır

Archery Men's Individual
Archery Women's Individual
Athletics Men's 10 kilometres Walk
Athletics Men's 10,000 metres
Athletics Men's 100 metres
Athletics Men's High Jump


"""




fig, axs = plt.subplots(2,2)
#fig.suptitle('Branşlara göre Kilo/Boy')

 
"""
Ploting = veriler_son.iloc[439:844,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
axs[0,0].plot(Ploting.iloc[0:405,1:2].values,Ploting.iloc[0:405,2].values,'r.')
axs[0,0].set_xlim([140, 220])  
axs[0,0].set_ylim([35, 110])
axs[0,0].set_title('Bayanlar Okçuluk') 
"""



Ploting = veriler_son.iloc[2613:3131,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
axs[0,0].plot(Ploting.iloc[0:518,1:2].values,Ploting.iloc[0:518,2].values,'g.')
axs[1,0].plot(Ploting.iloc[0:518,1:2].values,Ploting.iloc[0:518,2].values,'g.')
axs[0,0].set_xlim([140, 220])  
axs[0,0].set_ylim([35, 110]) 
axs[0,0].set_title('Yüksek Atlama') 

Ploting = veriler_son.iloc[1478:2613,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
axs[1,1].plot(Ploting.iloc[0:1135,1:2].values,Ploting.iloc[0:1135,2].values,'b.')
axs[1,1].set_xlim([140, 220])  
axs[1,1].set_ylim([35, 110]) 
#axs[0,0].set_title('100m Koşu') 

Ploting = veriler_son.iloc[844:1478,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
axs[0,1].plot(Ploting.iloc[0:634,1:2].values,Ploting.iloc[0:634,2].values,'y.')
axs[0,1].set_xlim([140, 220])  
axs[0,1].set_ylim([35, 110]) 
axs[0,1].set_title('Maraton 10Km') 


Ploting = veriler_son.iloc[844:1478,:]
Ploting.sort_values(by=['Height'], inplace=True)
print(Ploting)
axs[1,0].plot(Ploting.iloc[0:634,1:2].values,Ploting.iloc[0:634,2].values,'y.')
axs[1,0].set_xlim([140, 220])  
axs[1,0].set_ylim([35, 110]) 
#axs[0,1].set_title('Maraton 10Km') 


















#plt.xlim([120, 240])  
#plt.ylim([40, 200])  


plt.show()













"""
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

"""







#veri on isleme

"""
boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)
"""

#eksik veriler
"""
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
"""


"""
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)


#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

"""





