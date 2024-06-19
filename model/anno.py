import os
import pandas as pd
import numpy as np
path = "AffectNet/train_set/annotations"

aro_list=[]
val_list=[]
exp_list=[]
lnd_list=[]
for f in os.listdir(path):
    aro={}
    val={}
    exp={}
    lnd={}
    if f.endswith('aro.npy'):
        aro['filename']=f.split('_')[0]
        aro['Arousal'] = np.load(os.path.join(path,f))
        aro_list.append(aro)
    elif f.endswith('val.npy'):
        val['filename']=f.split('_')[0]
        val['Valance'] = np.load(os.path.join(path,f))
        val_list.append(val)
    elif f.endswith('exp.npy'):
        exp['filename']=f.split('_')[0]
        exp['Expression'] = np.load(os.path.join(path,f))
        exp_list.append(exp)
    # elif f.endswith('lnd.npy'):
    #     lnd['filename']=f.split('_')[0]
    #     lnd['Landmarks'] = np.load(os.path.join(path,f))
    #     lnd_list.append(lnd)
    

df_aro = pd.DataFrame(aro_list) 
df_val = pd.DataFrame(val_list) 
df_exp = pd.DataFrame(exp_list) 
# df_lnd = pd.DataFrame(lnd_list) 


df = pd.merge((pd.merge(df_aro,df_val,on="filename")), df_exp ,on="filename")
df.to_csv("AffectNet/train_annotation.csv")

    

