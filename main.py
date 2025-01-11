# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:39:04 2023

@author: chris
"""

import scipy
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd

#%% hand-craft PCA
import numpy as np

def pca(x, n_components):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_scaled = (x - x_mean) /1
    
    cov_matrix = np.cov(x_scaled, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:,idx]
    
    principal_components = eigenvectors[:,:n_components]
    
    pca_result = x_scaled @ principal_components
    
    return pca_result, eigenvalues, eigenvectors
    #explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
#%% hand-craft Naive Basis Classifier

class Gaussian_NB:
    def __init__(self):
        self.class_prior_ = None
        self.sigma_ = None
        self.mu_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        self.class_prior_ = np.zeros(n_classes)
        self.sigma_ = np.zeros((n_classes, n_features))
        self.mu_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_prior_[i] = X_c.shape[0] / n_samples
            self.sigma_[i, :] = np.var(X_c, axis=0)
            self.mu_[i, :] = np.mean(X_c, axis=0)

        return self

    def predict(self, X):
        n_samples, n_features = X.shape
        n_classes = len(self.class_prior_)

        p = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            p[:, i] = np.sum(-0.5 * np.log(2 * np.pi * self.sigma_[i, :])
                             - 0.5 * ((X - self.mu_[i, :]) ** 2) / self.sigma_[i, :],
                             axis=1) + np.log(self.class_prior_[i])

        return np.argmax(p, axis=1)

#%% Load the data
data = np.load('./ML_Train.npy')
test = np.load('./ML_Test.npy')
data_info = pd.read_csv('./ML_Train.csv')
test_info = pd.read_csv('./ML_Test.csv')

#%% Denoise training ECG signal
lead1  = np.zeros((12209,5000,1))

for i in range(12209):
    lead1[i,:,0] = data[i,0,:5000]
    
lead1_new = lead1[:,:,0]

fs = 500 
lowcut = 100 
order = 10 


b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)

sig_denoised = sig.filtfilt(b, a, lead1_new)

#%% find training data's r peak
import numpy as np
# Find the index of the maximum value in the signal
rpeak = []
for k in range(12209):
    max_peak_index = np.argmax(sig_denoised[k,:])
    
    max_peak_value = sig_denoised[k,[max_peak_index]]
  
    rpeak.append(max_peak_value[0])
rpeak = np.array(rpeak).reshape(-1,1)  

#%% get training data's sex info
sex = data_info.iloc[:,2].values
sex = np.array(sex).reshape(-1,1)

#%% Denoise testing ECG signal
test_lead1  = np.zeros((6000,5000,1))

for i in range(6000):
    test_lead1[i,:,0] = test[i,0,:5000]
    
test_lead1_new = test_lead1[:,:,0]

fs = 500 
lowcut = 100 
order = 4 


b, a = sig.butter(order, lowcut, 'lowpass', fs=fs)


test_sig_denoised = sig.filtfilt(b, a, test_lead1_new)

#%% find testing data's r peak
import numpy as np
# Find the index of the maximum value in the signal
rpeak_test = []
for k in range(6000):
    max_peak_index = np.argmax(test_sig_denoised[k,:])
    
    max_peak_value = test_sig_denoised[k,[max_peak_index]]
  
    rpeak_test.append(max_peak_value[0])
rpeak_test = np.array(rpeak_test).reshape(-1,1)  

#%% get testing data's sex info
sex_test = test_info.iloc[:,2].values
sex_test = np.array(sex_test).reshape(-1,1)

#%% change class name to number
data_label = data_info['Label']
data_label_num = np.zeros((12209,1))

for i in range(12209):
    if data_label[i] == 'NORM':
        data_label_num[i,0] = 0
    elif data_label[i] == 'MI':
        data_label_num[i,0] = 1
    elif data_label[i] == 'STTC':
        data_label_num[i,0] = 2
    elif data_label[i] == 'CD':
        data_label_num[i,0] = 3
        
#%% training above 40 .csv info
df_above_40 = pd.DataFrame()

df_above_40 = data_info[data_info['age'] >= 40]

#%% 
ab40_numlist = df_above_40['SubjectId'].tolist()


#%% get .npy (age above 40)
 
data_above_40 = np.zeros((10114,12,5000))

for i in range(len(ab40_numlist)):
    data_above_40[i,:,:] = data[ab40_numlist[i],:,:]
    
#np.save('ecg_above70.npy', data_above_70 )

#%% get label (age above 40)
label_above_40 =  np.zeros((10114,1))

for j in range(len(ab40_numlist)):
    label_above_40[j,:] = data_label_num[ab40_numlist[j],:]   

#%% rpeak (age above 40)
rpeak_ab40 = np.zeros(10114)
for i in range(len(ab40_numlist)):
    rpeak_ab40[i] = rpeak[ab40_numlist[i]]
#np.save('rpeak_ab40.npy',rpeak_ab40)
#%% sex (age above 40)
sex_ab40 = np.zeros(10114)
for i in range(len(ab40_numlist)):
    sex_ab40[i] = sex[ab40_numlist[i]]
#np.save('sex_ab40.npy',sex_ab40) 

#%% PCA of lead1 
ecg1_ab40 = data_above_40[:,0,:]
result1_ab40 = pca(ecg1_ab40,397)
pca1_ab40 = result1_ab40[0]
#np.save('PCA_lead1(ab40_397).npy',pca1_ab40)

#%% PCA of lead4
ecg4_ab40 = data_above_40[:,3,:]
result4_ab40 = pca(ecg4_ab40,397)
pca4_ab40 = result4_ab40[0]
#np.save('PCA_lead4(ab40_397).npy',pca4_ab40)  


### testing age above 40 
#%% testing above 40 .csv info
test_above_40 = pd.DataFrame()

test_above_40 = test_info[test_info['age'] >= 40]

#%% 
test_ab40_numlist = test_above_40['SubjectId'].tolist()
#%% 
 
test_above_40 = np.zeros((5411,12,5000))

for i in range(len(test_ab40_numlist)):
    test_above_40[i,:,:] = test[test_ab40_numlist[i],:,:]
    
#np.save('test_ecg_ab70.npy', test_above_70 )  
#%% test_rpeak (teat age above 40)
test_rpeak_ab40 = np.zeros(5411)
for i in range(len(test_ab40_numlist)):
    test_rpeak_ab40[i] = rpeak_test[test_ab40_numlist[i]]
#np.save('test_rpeak_ab40.npy',test_rpeak_ab40)
#%% test_sex (teat age above 40)
test_sex_ab40 = np.zeros(5411)
for i in range(len(test_ab40_numlist)):
    test_sex_ab40[i] = sex_test[test_ab40_numlist[i]]
#np.save('test_sex_ab40.npy',test_sex_ab40)


#%% PCA of lead1 
ecg1_ab40 = test_above_40[:,0,:]
result1_ab40 = pca(ecg1_ab40,397)
test_pca1_ab40 = result1_ab40[0]
#np.save('test_PCA_de_lead1(ab40_397).npy', test_pca1_ab40)
 
#%% PCA of lead1 
ecg4_ab40 = test_above_40[:,3,:]
result4_ab40 = pca(ecg4_ab40,397)
test_pca4_ab40 = result4_ab40[0]
#np.save('test_PCA_de_lead4(ab40_397).npy', test_pca4_ab40)    

### predict NB above 40 group
#%% training features
reshaped_rpeak_ab40 = rpeak_ab40.reshape(-1, 1)
reshaped_sex_ab40 = sex_ab40.reshape(-1, 1)

#%% 
comb = np.concatenate((pca1_ab40, pca4_ab40, reshaped_rpeak_ab40, reshaped_sex_ab40, label_above_40), axis=1)

np.random.shuffle(comb)

comb_data = np.delete(comb, -1, axis=1)
comb_label = comb[:,-1]

#%% testing features
reshaped_test_rpeak_ab40 = test_rpeak_ab40.reshape(-1, 1)
reshaped_test_sex_ab40 = test_sex_ab40.reshape(-1, 1)

#%% 
test_comb = np.concatenate((test_pca1_ab40, test_pca4_ab40, reshaped_test_rpeak_ab40, reshaped_test_sex_ab40), axis=1)

#%% using NB classifier to predict 
gnb = Gaussian_NB()
gnb.fit(comb_data, comb_label)
y_pred_ab40 = gnb.predict(test_comb)


###### <40
#%%
df_under_40 = pd.DataFrame()

df_under_40 = test_info[(40 > test_info['age'])]


#%%
test_un40_numlist = df_under_40['SubjectId'].tolist()
#%% 
test_un40_array = np.array(test_un40_numlist)
result_un40_array = np.zeros((589, 2))
result_un40_array[:, 0] = test_un40_array

#%% integrate result
import numpy as np

y_pred_ab40_reshaped = y_pred_ab40.reshape(-1, 1)

combined_ab40 = np.column_stack((test_ab40_numlist, y_pred_ab40_reshaped))

#%%
prediction = np.concatenate((combined_ab40,result_un40_array), axis = 0)

sorted_indices = np.argsort(prediction[:, 0])

sorted_prediction = prediction[sorted_indices]


#%% save the prediction result in .csv
output = pd.DataFrame(sorted_prediction)
output.columns = ['SubjectId', 'Label']
output.to_csv('output_L1L4_rpeak_sex(h40(0))_sub.csv', index=False, header = True, float_format='%d')

   



















