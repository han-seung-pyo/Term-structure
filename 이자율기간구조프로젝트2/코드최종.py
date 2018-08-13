# -*- coding: utf-8 -*-
"""
Created on Sat May 26 05:14:14 2018

@author: 삼성컴퓨터
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:44:28 2018

@author: rbgud
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

#날짜 처리
td = (raw_ts.index - dt.datetime.strptime('2018-05-24','%Y-%m-%d'))
x = (td / np.timedelta64(1, 'M')).astype(float)
y = raw_ts['Zero Rate']
#보간법
lx = x
ly = np.exp(-y*lx/12)
lf = spi.interp1d(lx,ly, kind = 'cubic')


xnew = pd.DataFrame(np.arange(1,121,1))
ynew = pd.DataFrame(lf(xnew))

#plt.plot(x,y,'r.',label='zero rate')
plt.plot(xnew.loc[0:120],ynew.loc[0:120],'b-',label='interpolate')
plt.show()

# vol > std 변환 
empty = pd.DataFrame()
for i in range(10):
    aa = raw_vol.iloc[4+i,9-i]
    bb = pd.Series([aa]*12)
    empty = pd.concat([empty,bb])
sigma = np.sqrt(np.array(empty))/10000

    

#%%

#Discount factor
dsc =np.array( ynew.loc[0:120])
t = np.arange(1,121,1)
spot = np.array(-12*np.log(dsc)/xnew.loc[0:120])

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
    tree.append(np.linspace(r0 + i*sigma[i]*np.sqrt(dt), r0 - i*sigma[i]*np.sqrt(dt),i+1))  #theta 없음
    drift = Theta(dsc[i], tree, Q, dt)
    theta.append(drift)
    tree[-1] += drift


#%%
lxx = np.log(x)
lyy = np.log(y)
lf = spi.interp1d(lxx,lyy, kind = 'linear')

xnew1 = pd.DataFrame(np.arange(1,121,1))
ynew1 = pd.DataFrame(np.exp(lf(np.log(xnew1))))

#plt.plot(x,y,'r.',label='zero rate')
plt.plot(xnew1.loc[0:120],ynew1.loc[0:120],'b-',label='interpolate')
plt.show()

#Discount factor
t2 = np.arange(1,121,1)
spot2 = np.exp(lf(np.log(t2)))
dsc2 = np.exp(-spot2 * t2/12)
    
empty2 = pd.DataFrame()
for i in range(10):
    aa = raw_vol.iloc[4+i,9-i]
    bb = pd.Series([aa])
    empty2 = pd.concat([empty2,bb])

vol2 = np.array(empty2)    


empty3 = pd.DataFrame([vol2[0]]*12)
for i in range(0,9):
    aaa = ((i+1)*(vol2[i+1][0])) - ((i)*(vol2[i][0]))
    bbb = pd.Series([aaa]*12)
    empty3 = pd.concat([empty3,bbb])
    
f_vol = np.sqrt(np.array(empty3))/10000

#%%

r0_1 = spot2[0]
tree2 = [np.array([r0_1])]
Q2 = [np.array([1])]
theta = [0]

#Tree building
for i in range(1,len(t)):
    Qi = np.zeros(i+1)
    for j in range(i):
        Qi[j] += p * Q2[i-1][j] * np.exp(-tree2[i-1][j]*dt)
        Qi[j+1] += (1-p) * Q2[i-1][j] * np.exp(-tree2[i-1][j]*dt)
    Q2.append(Qi)
    tree2.append(np.linspace(r0 + i*f_vol[i]*np.sqrt(dt), r0 - i*f_vol[i]*np.sqrt(dt),i+1))  #theta 없음
    drift2 = Theta(dsc2[i], tree2, Q2, dt)
    theta.append(drift2)
    tree2[-1] += drift2
    