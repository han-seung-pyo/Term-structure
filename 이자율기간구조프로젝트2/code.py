# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:46:33 2018

@author: 삼성컴퓨터
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import datetime as dt
import os
os.chdir('C:\\Users\삼성컴퓨터\Desktop\이자율기간구조프로젝트2')
#데이터 가지고 오기
raw_ts = pd.read_excel('0524_libor_vol.xlsx',sheetname='Term_Structure')
raw_vol = pd.read_excel('0524_libor_vol.xlsx',sheetname='vol').set_index('만기')
raw_ts.index = raw_ts.pop('Maturity Date')
raw_ts.index = pd.DatetimeIndex(raw_ts.index)
raw_ts['Zero Rate'] = raw_ts['Zero Rate']/100

empty = pd.DataFrame()
for i in range(10):
    aa = raw_vol.iloc[4+i,9-i]
    bb = pd.Series([aa]*12)
    empty = pd.concat([empty,bb])
sigma = np.array(empty)

#날짜 처리
td = (raw_ts.index - dt.datetime.strptime('2018-05-24','%Y-%m-%d')) #일별 날짜 처리
x = (td / np.timedelta64(1, 'M')).astype(float) #월별로 날짜 처리 1이 1달 2.18은 2달하고도 18일
y = raw_ts['Zero Rate'] 

#보간법
lx = x
ly = np.exp(-y*lx/12) #discount factor로 만들기
lf = spi.interp1d(lx,ly, kind = 'cubic') #interpolate 돌리기

#120개월을 만들어서 interpolate 하기
xnew = pd.DataFrame(np.arange(1,121,1))
ynew = pd.DataFrame(lf(xnew))


plt.plot(xnew.loc[0:120],ynew.loc[0:120],'b-',label='interpolate')
plt.show()


#Discount facor 및 spot rate
dsc = np.array(ynew.loc[0:120]) #120개에 대한 discount facotor
t = np.arange(1,121,1) 
spot = np.array(-12*np.log(dsc)/xnew.loc[0:120]) #spot rate로 변환
#%%
#Ho-Lee model
sigma1 = 0.01
dt = 1/12
p = 0.5
r0 = spot[0]
tree = [np.array([r0])]
Q = [np.array([1])]
theta = [0]

def Theta(dsc, tree, Q, dt):
    dsc_unadj = (np.exp(-tree[-1]*dt) * Q[-1]).sum()
    return (-1/dt)*np.log(dsc / dsc_unadj)

#Tree building
for i in range(1,len(t)):
    Qi = np.zeros(i+1)
    for j in range(i):
        Qi[j] += p * Q[i-1][j] * np.exp(-tree[i-1][j]*dt)
        Qi[j+1] += (1-p) * Q[i-1][j] * np.exp(-tree[i-1][j]*dt)
    Q.append(Qi)
    tree.append(np.linspace(r0 + i*sigma1*np.sqrt(dt), r0 - i*sigma1*np.sqrt(dt),i+1))  #theta 없음
    drift = Theta(dsc[i], tree, Q, dt)
    theta.append(drift)
    tree[-1] += drift

tree1 = [np.array([r0])]  
    
def treeselect(st,node):
    select_tree = [np.array([tree[st][node]])]
    for i in range(1,len(t)-st):
        select_tree.append(tree[st+i][node:node+i+1])
    return select_tree 

Q = [np.array([1])]
for i in range(12,len(t)):
    Qi1 = np.zeros(i-10)
    for j in range(i-11):
        Qi1[j] += p * Q[i-12][j] * np.exp(-treeselect(12,0)[i-12][j] * dt)
        Qi1[j+1] = (1-p) * Q[i-12][j] * np.exp(-treeselect(12,0)[i-12][j] * dt)
    Q.append(Qi1)
    tree1.append(np.linspace(r0 + i*sigma[i]*np.sqrt(dt), r0 - i*sigma[i]*np.sqrt(dt),i+1))
    drift1 = Theta(dsc[i], tree1, Q, dt)
    theta.append(drift1)
    tree1[-1] += drift1