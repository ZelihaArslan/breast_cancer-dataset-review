# -*- coding: utf-8 -*-



from matplotlib import pyplot as plt
import kutuphane
from sklearn.feature_selection import SelectKBest, f_classif

import warnings
warnings.filterwarnings("ignore")

#1- Verileri yükle
giris, cikis, kisi_bilgisi =  kutuphane.dosya_oku('breast-cancer-wisconsin.csv')

#2 Ölçeklendir
olcekli_giris = kutuphane.olceklendir(giris)

#Parametre Optimizasyonu
sonuclar = kutuphane.parametre_optimizasyonu(giris,cikis, kisi_bilgisi)

#Özellik Seçimi
dogruluk_chi = []

#Chi-square 
for k in range(10,500,10):
    ozellikler  = kutuphane.chi2_ozellik_cikar(giris,cikis,k)
    azaltilmis_olcekli_giris = olcekli_giris[:,ozellikler]
    dogruluk,f1skor = kutuphane.basari_hesaplaCV(azaltilmis_olcekli_giris, cikis, kisi_bilgisi,10)
    dogruluk_chi.append(dogruluk)
    print("k="+str(k) + " acc="+str(dogruluk))

dogruluk_anova = []
for k in range(10,150,10):
    azaltilmis_olcekli_giris = SelectKBest(f_classif, k=k).fit_transform(X=olcekli_giris,y=cikis)
    dogruluk,f1skor = kutuphane.basari_hesapla(azaltilmis_olcekli_giris, cikis, kisi_bilgisi)
    print("k="+str(k) + " acc="+str(dogruluk))
    dogruluk_anova.append(dogruluk)

import numpy as np
x_ekseni = np.arange(10,500,10)
fig,plots = plt.subplots(2,1)
plots[0].plot(x_ekseni,dogruluk_chi)
plots[0].set_xlabel('Chi-square Özellik Sayısı')
plots[0].set_ylabel('Doğruluk')
plots[1].plot(dogruluk_anova)
plots[1].set_xlabel('Anova-F Özellik Sayısı')
plots[1].set_ylabel('Doğruluk')

plt.show()