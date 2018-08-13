#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 17:52:43 2018

@author: 삼성컴퓨터
"""

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import scipy as sp
import os
os.chdir('C:\\Users\삼성컴퓨터\Desktop\대학원수업\이자율기간구조\프로젝트')

df=pd.read_excel(r'interest_hw_data.xlsx', sheetname = 'Sheet3', header=0, index_col=0)

x0 = [0,0,0,0,0,0] #b0, b1, b2, tow1, tow2
#함수 Rm
def rm1(list): #list=[0.1,0.1,0.1,0.1,1,1] b0,b1,b2,b3,tow1,tow2
    diff_list = []
    for i in range(0,len(df)):
        t1=list[0]
        t2=list[1]*(1-np.exp(-df['tillMat'][i]/list[4]))/(df['tillMat'][i]/list[4])
        t3=list[2]*((1-np.exp(-df['tillMat'][i]/list[4]))/(df['tillMat'][i]/list[4])-np.exp(-df['tillMat'][i]/list[4]))
        t4=list[3]*((1-np.exp(-df['tillMat'][i]/list[5]))/(df['tillMat'][i]/list[5])-np.exp(-df['tillMat'][i]/list[5]))
        x=t1+t2+t3+t4
        
        start=df['start'][i]
        price=df['PX_LAST'][i]
        m=df['tillMat'][i]
        coupon = df['CPN'][i]/2
        phi=df['DUR_ADJ_MID'][i]*df['PX_LAST'][i]/(1+df['YLD_YTM_MID'][i]/100)
        f=[]
        for k in range(0,1+int(((m-start)*2))):
            q= start +( 0.5 * k)
            f.append(np.exp(-x*q/100))
        pm = sum(f) * coupon + np.exp(-x*m/100)*100
        diff=np.power((df['PX_LAST'][i]-pm)/phi,2)
        diff_list.append(diff)
    return sum(diff_list)
#solver
a=sp.optimize.minimize(rm1,x0,method='Nelder-Mead', tol=1e-10)


#FITTING
p=a['x']
df1=pd.DataFrame(np.linspace(0,30,101), columns=['tillMat'])
Svensson=[]
for i in np.arange(0,len(df1)):
        t1=p[0]
        t2=p[1]*(1-np.exp(-df1['tillMat'][i]/p[4]))/(df1['tillMat'][i]/p[4])
        t3=p[2]*((1-np.exp(-df1['tillMat'][i]/p[4]))/(df1['tillMat'][i]/p[4])-np.exp(-df1['tillMat'][i]/p[4]))
        t4=p[3]*((1-np.exp(-df1['tillMat'][i]/p[5]))/(df1['tillMat'][i]/p[5])-np.exp(-df1['tillMat'][i]/p[5]))
        x=(t1+t2+t3+t4)/100
        Svensson.append(x)
Svensson=pd.DataFrame(Svensson)

#Nelson and Siegel method
x1=[0,0,0,0] # b0, b1, b2,tow1
#함수
def rm2(list):
    diff_list = []
    for i in range(0,len(df)):
        t1=list[0]
        t2=list[1]*(1-np.exp(-df['tillMat'][i]/list[3]))/(df['tillMat'][i]/list[3])
        t3=list[2]*((1-np.exp(-df['tillMat'][i]/list[3]))/(df['tillMat'][i]/list[3])-np.exp(-df['tillMat'][i]/list[3]))
        x=t1+t2+t3
        start=df['start'][i]
        price=df['PX_LAST'][i]
        m=df['tillMat'][i]
        coupon = df['CPN'][i]/2
        phi=df['DUR_ADJ_MID'][i]*df['PX_LAST'][i]/(1+df['YLD_YTM_MID'][i]/100)
        f=[]
        for k in range(0,1+int(((m-start)*2))):
            q= start +( 0.5 * k)
            f.append(np.exp(-x*q/100))
        pm = sum(f) * coupon + np.exp(-x*m/100)*100
        diff=np.power((df['PX_LAST'][i]-pm)/phi,2)
        diff_list.append(diff)
    return sum(diff_list)
#solver
b=sp.optimize.minimize(rm2,x1,method='Nelder-Mead', tol=1e-10)

#FITTING
p1=b['x']
NS=[]
for i in np.arange(0,len(df1)):
        t1=p1[0]
        t2=p1[1]*(1-np.exp(-df1['tillMat'][i]/p1[3]))/(df1['tillMat'][i]/p1[3])
        t3=p1[2]*((1-np.exp(-df1['tillMat'][i]/p1[3]))/(df1['tillMat'][i]/p1[3])-np.exp(-df1['tillMat'][i]/p1[3]))
        x=(t1+t2+t3)/100
        NS.append(x)
NS=pd.DataFrame(NS)
print(NS)
#plot
plt.plot(df1['tillMat'],NS, label='NS')
plt.plot(df1['tillMat'],Svensson, label='SV')
plt.legend(loc='upper left')
plt.plot(df['tillMat'],df['YLD_YTM_MID']/100, 'gs')
plt.show()
