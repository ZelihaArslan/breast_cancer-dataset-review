import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import LeaveOneGroupOut,KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def dosya_oku(dosyaAdi):
    file_data = pd.read_csv(dosyaAdi)
    group_ids=file_data.iloc[:,0]
    raw_data=file_data.iloc[:,1:-1]
    result_data=file_data.iloc[:,-1]
    return raw_data, result_data, group_ids



def parametre_optimizasyonu(giris, cikis, kisi_bilgisi):

    siniflandirici = SVC()
    egitim_giris, test_giris,egitim_cikis, test_cikis = train_test_split(giris, cikis, test_size=0.8, random_state=30)
    egitim_giris = olceklendir(egitim_giris)

    parametreler = [ {'kernel':['linear'], 'C':[1,10,20,50,100]},
                    {'kernel':['rbf'], 'C':[1,10,20,50,100], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] }]
    #parametreler = [ {'criterion':['gini','entropy'], 'n_estimators':np.arange(10,101,10) }, {'criterion':['gini','entropy'], 'n_estimators':np.arange(10,101,10) } ]
    #parametreler = [{'n_neighbors':[4,5,6], 'metric':['minkowski','chebyshev','euclidean'] }]
    
    
    arayici = GridSearchCV(estimator=siniflandirici, 
                           param_grid=parametreler,
                           scoring='accuracy',
                           #cv=int(len(egitim_giris)/3) )
                           cv=3)    
    
    arayici = arayici.fit(egitim_giris,egitim_cikis)
    sonucs = arayici.grid_scores_ 
    
   
    print('en yuksek basari:'+ str(arayici.best_score_) )
    print('en iyi parametreler:'+ str(arayici.best_params_) )
    return sonucs
    
    
    
    
def basari_hesaplaCV(giris,cikis, kisi_bilgisi,cv=252):
    folder = KFold(n_splits=cv)
    
    #Destek vektör sınıflandırıcısı
    clf=SVC(kernel='linear',C=1.0)
    
    toplamBasari = 0
    toplamFSkor = 0
    
    for train_index, test_index in folder.split(giris, cikis, kisi_bilgisi):
        #Eğitim ve test verilerini ayır
        X_train, X_test = giris[train_index,:], giris[test_index,:]
        y_train, y_test = cikis.iloc[train_index], cikis.iloc[test_index]           
        
        #Modeli eğit.
        clf.fit(X_train, y_train) 
        #Modelden tahmin iste.
        pred_y = clf.predict(X_test)        
        
        #Tahminlerin başarılarını hesapla.
        toplamBasari +=  accuracy_score(y_test,pred_y)
        toplamFSkor  += f1_score(y_test,pred_y)  
    #Ortalama Başarı = toplam başarı / parça sayısı
    return toplamBasari/cv, toplamFSkor/cv

def basari_hesapla(giris,cikis, kisi_bilgisi):
    #Kişi bazlı çapraz doğrulama
    logo = LeaveOneGroupOut()
    #Destek vektör sınıflandırıcısı
    clf=SVC(C=1, gamma=0.2, kernel='rbf')
    #clf = RandomForestClassifier(criterion='entropy',n_estimators=60)
    toplamBasari = 0
    toplamFSkor = 0
    
    for train_index, test_index in logo.split(giris, cikis, kisi_bilgisi):
        #Eğitim ve test verilerini ayır
        X_train, X_test = giris[train_index,:], giris[test_index,:]
        y_train, y_test = cikis.iloc[train_index], cikis.iloc[test_index]           
        
        #Modeli eğit.
        clf.fit(X_train, y_train) 
        #Modelden tahmin iste.
        pred_y = clf.predict(X_test)        
        
        #Tahminlerin başarılarını hesapla.
        toplamBasari +=  accuracy_score(y_test,pred_y)
        toplamFSkor  += f1_score(y_test,pred_y)  
    #Ortalama Başarı = toplam başarı / parça sayısı
    return toplamBasari/logo.get_n_splits(giris, cikis, kisi_bilgisi), toplamFSkor/logo.get_n_splits(giris, cikis, kisi_bilgisi)


def chi2_ozellik_cikar(giris, cikis, ozellik_sayisi):
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(giris)
    chi_vals, p_vals = chi2(X,cikis)
    ozellikler  = np.argsort(chi_vals)[::-1][:ozellik_sayisi]
    return ozellikler


def olceklendir(giris):
    olceklendirici = StandardScaler()
    return olceklendirici.fit_transform(giris)