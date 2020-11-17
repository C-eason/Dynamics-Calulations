#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# @File: cutoff-id.py
# @Author:xiaoqian
# @Time: 2020.11.15.16:02
# @comments:
'''
import pandas as pd
import numpy as np
import json

def read(a):
    size=[]
    Side_len = {}
    with open (a,'r') as file:
        f=file.readlines()
        natom=int(f[3])

        s=f[5].split()
        size.append(float(s[1])-float(s[0]))
        Side_len['xlo'] =  float(s[0])
        Side_len['xhi'] = float(s[1])
        s = f[6].split()
        size.append(float(s[1]) - float(s[0]))
        Side_len['ylo'] = float(s[0])
        Side_len['yhi'] = float(s[1])
        s = f[6].split()
        size.append(float(s[1]) - float(s[0]))
        Side_len['zlo'] = float(s[0])
        Side_len['zhi'] = float(s[1])
        nsize=np.array(size)
        hmatrix=np.diag(nsize)

        name=f[8].split()[2:]
    data=pd.read_csv(a,delim_whitespace=True,skiprows=9,names=name,header=None,nrows=natom)
    # data.sort_values(by='id',inplace=True)
    return natom,nsize,Side_len,hmatrix,data

def neighbor(cutoff=2.92, filepath="1900K-cfg.lammpstrj"):
    natom, nsize, Side_len, hmatrix, data=read(filepath)
    Al=data["type"]==2
    Al_df=data[Al]
    Al_position=Al_df.iloc[:,2:].values
    Al_id=Al_df.iloc[:,0].values
    hmatrixinv = np.linalg.inv(hmatrix)
    inner_id=[]
    outer_id=[]
    for i in range(Al_position.shape[0]):
        RIJ=Al_position-Al_position[i]
        matrixij = np.dot(RIJ, hmatrixinv)
        RIJ = np.dot(matrixij - np.rint(matrixij) * [1,1,1], hmatrix)  # remove PBC
        RIJ_norm = np.linalg.norm(RIJ, axis=1)
        nearests = Al_id[RIJ_norm <= cutoff]
        X_id=nearests.tolist()
        if len(X_id)==1:
            outer_id.append(X_id[0])
        else:
            for x in X_id:
                inner_id.append(x)

    ID = list(set(inner_id))
    print(len(ID),len(outer_id))
    IDstr = [str(x) for x in ID]
    outerstr=[str(x) for x in outer_id]
    dic={"inner":IDstr,"outer":outerstr}
    filename='al.json'
    with open(filename,'w') as file_obj:
        json.dump(dic,file_obj)

neighbor(cutoff=3.46,filepath="1900K-cfg.lammpstrj")
