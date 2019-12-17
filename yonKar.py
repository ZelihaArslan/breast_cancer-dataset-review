# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 04:13:09 2019

@author: Hatice Sahin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


veri = pd.read_csv('breast-cancer-wisconsin.csv')
ozellik_sayısı = 9


# Giriş-Çıkış Belirle
giris_verileri = veri.iloc[:,2:ozellik_sayısı+1]
cıkıs = veri.iloc[:,-1]


# Eğitim ve Test Verilerini Belirle
egitim_giris,test_giris,egitim_cikis,test_cikis = train_test_split(giris_verileri,cıkıs,test_size=0.15,random_state=0)


#Standardizasyon
scaler = preprocessing.StandardScaler()
stdGiris = scaler.fit_transform(egitim_giris)
stdTest = scaler.transform(test_giris)

siniflandiricilar=[KNeighborsClassifier(n_neighbors=4), LogisticRegression(random_state=0), GaussianNB(), DecisionTreeClassifier(), SVC(), RandomForestClassifier(n_estimators=40)]


basari=list()
fSkor = list()

for i in range(6):
    
    siniflandiricilar[i].fit(stdGiris, egitim_cikis)
    cikis_tahmin = siniflandiricilar[i].predict(stdTest)
    print("\n")
    print(confusion_matrix(test_cikis, cikis_tahmin))
    basari.append(accuracy_score(test_cikis, cikis_tahmin))
    fSkor.append( f1_score(test_cikis, cikis_tahmin, labels=None, pos_label=1, average='binary', sample_weight=None))


