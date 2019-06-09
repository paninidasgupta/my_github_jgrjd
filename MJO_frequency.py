## phase calculation using PC1 and PC2 values

import numpy as np
import scipy as sc
from numpy import linalg as LA

# %matplotlib notebook

def theta1_cal( pc1, pc2 ):
    theta1           =  np.zeros(pc2.size)
    angle180         =  np.arctan2(-pc2, -pc1) * 180 / np.pi
    indNeg           =  np.where(angle180<0)[0]
    angle360         = angle180
    angle360[indNeg] = angle180[indNeg] + 360
    theta1           = np.around( ( angle360 + 22.5 ) * 8 / 360 )

    
    return theta1
 
        

#%% phase data in extended winter of each year, should provide no. of year, start year and starting index.

def theta1_winter(theta1,amp,year,ny,start_index):
    the1_wintr    =   np.zeros((ny,4), dtype=np.object)
    amp_wintr     =   np.zeros((ny,4),dtype=np.object)
    mon_s1        =   [90,92,92,91];
    no_d1         =   [90,92,92,91];
    mon_s2        =   [91,92,92,91];
    no_d2         =   [91,92,92,91];
    for i in range(ny):
        for j in range(4):
            if ((year)%4==0):
                the1_wintr[i][j] =  theta1[start_index:start_index+no_d2[j]]
                amp_wintr[i][j]  =  amp[start_index:start_index+no_d2[j]]
                start_index      =  start_index+mon_s2[j]
            else:
                the1_wintr[i][j] =  theta1[start_index:start_index+no_d1[j]]
                amp_wintr[i][j]  =  amp[start_index:start_index+no_d1[j]]
                start_index      =  start_index+mon_s1[j]
            
        year      =     year+1
            
            
    return the1_wintr,amp_wintr

def select_seg(a):
    new_arr= a*1
    prev = 0
    splits = np.append(np.where(np.diff(a) != 0)[0],len(a)+1)+1
    t=0
    w=[]
    seg=[]
    for split in splits:
        #print (np.arange(0,a.size,1)[prev:split])
        w.append(list(np.arange(0,a.size,1)[prev:split]))
        prev = split
        t=t+1
        
    for i in np.arange(len(w)):
        if len(w[i])<10:
            new_arr[w[i]]=0
        elif not np.any(new_arr[w[i]]):
            continue
        else:
            seg.append(w[i])
    
    
    return new_arr,seg


#%% Finding the phase occurances in each year

def phase_occurances(the1_obj,amp_obj,thers_amp,opt):
    
    ## ** opt=0 for amplitude || opt==1 for phase occurances**###
    
    to_dur             = np.zeros((the1_obj.shape[0],the1_obj.shape[1],8))
    ano_to_dur         = np.zeros((the1_obj.shape[0],the1_obj.shape[1],8))

    ano_to_dur_djf     = np.zeros((the1_obj.shape[0],8))
    ano_to_dur_mam   = np.zeros((the1_obj.shape[0],8))
    ano_to_dur_jja   = np.zeros((the1_obj.shape[0],8))
    ano_to_dur_son  = np.zeros((the1_obj.shape[0],8))

    for i in range(the1_obj.shape[0]):
        for j in range(the1_obj.shape[1]):
            th         =  the1_obj[i][j]
            aa         =   amp_obj[i][j]
            a_temp      =   (aa>thers_amp)*1
            new_arr,seg =   select_seg(a_temp)
            for k in range(8):
                ind_f            =  np.where(th==k+1)[0]
                ind_s            =  np.where(new_arr==1)[0]
                values           =  np.intersect1d(ind_f,ind_s)
                if np.any(len(values)):
                    if opt:
                        to_dur[i][j][k]  =  values.size
                    else:
                        temp=  np.nanmean(aa[values])
                        if np.isnan(temp): temp=0
                        to_dur[i][j][k]  =  temp
                else:
                        to_dur[i][j][k]  =  0

    
    for i in range(to_dur.shape[1]):
        ano_to_dur[:,i,:]    = (to_dur[:,i,:]-np.nanmean(to_dur[:,i,:],axis=0))
        
    ano_to_dur_djf    = ano_to_dur[:,0,:]
    ano_to_dur_mam    = ano_to_dur[:,1,:]
    ano_to_dur_jja    = ano_to_dur[:,2,:]
    ano_to_dur_son   = ano_to_dur[:,3,:]

    return     ano_to_dur_djf, ano_to_dur_mam, ano_to_dur_jja, ano_to_dur_son,to_dur
            

        
#%% EOF analysis of the phase occurance data 

def eof_wrap(matrix):
    ny            =         matrix.shape[0]
    mat           =         np.matrix(matrix)
    mean          =         mat.mean(0)
    std           =         mat.std(0)
    one           =         np.matrix(np.ones(ny,float))
    mean_matrix   =         one.T*mean
    std_matrix    =         one.T*std
    anomaly11     =         mat-mean_matrix
    anomaly22     =         anomaly11/std
    cov_matrix    =         anomaly11.T*anomaly11
    D,V           =         LA.eig(cov_matrix)
    varex         =         D/np.sum(D)
    max_eig_ind   =         np.where(varex==np.max(varex))
    sec_eig_ind   =         np.where(varex==np.max(varex[varex<np.max(varex)]))
    third_eig_ind =         np.where(varex==np.max(varex[varex<varex[sec_eig_ind]]))
    fourth_eig_ind=         np.where(varex==np.max(varex[varex<varex[third_eig_ind]]))
    EOF1          =         V[:,max_eig_ind]
    EOF2          =         V[:,sec_eig_ind]
    EOF3          =         V[:,third_eig_ind]
    EOF4          =         V[:,fourth_eig_ind]
    PC1           =         anomaly11*np.squeeze(EOF1).T
    PC2           =         anomaly11*np.squeeze(EOF2).T
    PC3           =         anomaly11*np.squeeze(EOF3).T
    PC4           =         anomaly11*np.squeeze(EOF4).T
    reconstructed =         PC1*(np.squeeze(EOF1))+PC2*(np.squeeze(EOF2))+PC3*(np.squeeze(EOF3))
    return anomaly11,D,V,cov_matrix,EOF1,EOF2,EOF3,EOF4,varex,PC1,PC2,PC3,PC4,reconstructed

#%% testing significance of EOFs

def test_significance(D,ny):
    DD          =         np.sort(D)
    varex       =         DD/np.sum(DD)
    significant =         np.zeros(DD.size,float)  
    error =         np.zeros(DD.size,float)  
    k           =         0
    for i in range (DD.size):
         D_new  =         abs(DD-DD[i])
         error[i]  =         DD[i]*(2/ny)**0.5
         k      =         k+1
         if (sum(D_new>error[i])==DD.size-1):
             significant[i]=1
         else:
             significant[i]=0
        
    if significant[DD.size-1]==1:
        print("EOF1 is significant and pressenting variance %f" %varex[DD.size-1])
    else:
        print("EOF1 is not significant and pressenting variance %f" %varex[DD.size-1])

    if significant[DD.size-2]==1:
        print("EOF2 is significant and presenting variance %f" %varex[DD.size-2])
    else:
        print("EOF2 is not significant and pressenting variance %f" %varex[DD.size-2])
    
    if significant[DD.size-3]==1:
        print("EOF3 is significant and pressenting variance %f" %varex[DD.size-3])
    else:
        print("EOF3 is not significant and pressenting variance %f" %varex[DD.size-3])
    
    if significant[DD.size-4]==1:
        print("EOF4 is significant and pressenting variance %f" %varex[DD.size-4])
    else:
        print("EOF4 is not significant and pressenting variance %f" %varex[DD.size-4])

    
    
    return k,DD,significant,error,varex      