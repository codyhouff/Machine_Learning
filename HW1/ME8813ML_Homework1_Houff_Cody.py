#####################################################
##  ME8813ML Homework 1: 
##  Implement a quasi-Newton optimization method for data fitting
#####################################################
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols,diff
import sympy as sp
import numpy as np
from scipy.misc import derivative
import math


########################################################
## Implement a parameter fitting function fit() so that 
##  p = DFP_fit(x,y)
## returns a list of the parameters as p of model:
##  p0 + p1*cos(2*pi*x) + p2*cos(4*pi*x) + p3*cos(6*pi*x) + p4*cos(8*pi*x) 
########################################################


# Fixing random state for reproducibility
np.random.seed(19680801)

dx = 0.1
x_lower_limit = 0
x_upper_limit = 40                                       
x = np.arange(x_lower_limit, x_upper_limit, dx)
data_size = len(x)                                 # data size
noise = np.random.randn(data_size)                 # white noise

# Original dataset 
y = 2.0 + 3.0*np.cos(2*np.pi*x) + 1.0*np.cos(6*np.pi*x) + noise

#print("x_array: ",x_array)
#print("y_array: ",y_array)

###########################################
def partial_derivative():
    
    x, y, p0, p1, p2, p3 = symbols('x y p0, p1, p2, p3', real=True)
    f = (p0 + p1*sp.cos(2*sp.pi*x) + p2*sp.cos(4*sp.pi*x) + p3*sp.cos(6*sp.pi*x)-y)**2

    df_p0 = diff(f, p0)
    df_p1 = diff(f, p1)
    df_p2 = diff(f, p2)
    df_p3 = diff(f, p3)
    print(df_p0)
    print(df_p1)
    print(df_p2)
    print(df_p3)
    print(" ")


def calc_f(x,y,p):
    f = (p[0,0] + p[1,0]*np.cos(2*np.pi*x) + p[2,0]*np.cos(4*np.pi*x) + p[3,0]*np.cos(6*np.pi*x)-y)**2
    f = np.sum(f)
    return f

def calc_df(x,y,p):
    df_p0 = 2*p[0,0] + 2*p[1,0]*np.cos(2*np.pi*x) + 2*p[2,0]*np.cos(4*np.pi*x) + 2*p[3,0]*np.cos(6*np.pi*x) - 2*y
    df_p1 = 2*(p[0,0] + p[1,0]*np.cos(2*np.pi*x) + p[2,0]*np.cos(4*np.pi*x) + p[3,0]*np.cos(6*np.pi*x) - y)*np.cos(2*np.pi*x)
    df_p2 = 2*(p[0,0] + p[1,0]*np.cos(2*np.pi*x) + p[2,0]*np.cos(4*np.pi*x) + p[3,0]*np.cos(6*np.pi*x) - y)*np.cos(4*np.pi*x)
    df_p3 = 2*(p[0,0] + p[1,0]*np.cos(2*np.pi*x) + p[2,0]*np.cos(4*np.pi*x) + p[3,0]*np.cos(6*np.pi*x) - y)*np.cos(6*np.pi*x)

    df = np.array([[np.sum(df_p0)], [np.sum(df_p1)], [np.sum(df_p2)], [np.sum(df_p3)]])
    return df

def DFP_fit(x_array,y_array,e):

    B = np.identity(4)

    p = np.array([[4], [4], [4], [4]])

    df = calc_df(x,y,p)
    #for x, y in zip(x_array, y_array):
    while(np.linalg.norm(df)**2>e):

        df = calc_df(x,y,p)
        d = -B@df
        f = calc_f(x,y,p)
        #f = np.sum(f)

        alpha = 1
        beta = .3 #5
        while(True):
            alpha = beta*alpha  
            p_temp = p - alpha*df
            f_temp = calc_f(x,y,p_temp)
            f_temp = np.sum(f_temp)
            #f = calc_f(x,y,p)
            compare = e*alpha*(np.linalg.norm(p))**2

            if(f-f_temp<=compare):
                break

        p_new = p+alpha*d
        dp = p_new - p
        p = p_new
        df_new = calc_df(x,y,p_new)
        dg = df_new - df

        #if (np.linalg.norm(dg)<e):
        #print("p: ",p)
        #return p

        B = B+ (dp@dp.T)/(dp.T@dg) - ((B@dg)@(B@dg).T)/((dg.T@B)@dg)

    print(p)
    return p


def xy_predict(x_array,y_array,p):

    x_predict = []
    y_predict = []                    
    for x, y in zip(x_array, y_array):    
        y = p[0,0] + p[1,0]*np.cos(2*np.pi*x) + p[2,0]*np.cos(4*np.pi*x) + p[3,0]*np.cos(6*np.pi*x)
        x_predict.append([x])
        y_predict.append([y])

    x_predict = np.array(x_predict)
    y_predict = np.array(y_predict)

    return x_predict, y_predict

e = .000006
partial_derivative()
p = DFP_fit(x,y,e)
x_predict, y_predict = xy_predict(x,y,p)

###########################################


fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y)
axs[0].set_xlim(x_lower_limit, x_upper_limit)
axs[0].set_xlabel('x')
axs[0].set_ylabel('observation')
axs[0].grid(True)


#########################################
## Plot the predictions from your fitted model here
#axs[0].plot(x_predict, y_predict)
axs[1].plot(x_predict, y_predict)
axs[1].set_xlim(x_lower_limit, x_upper_limit)
axs[1].set_xlabel('x')
axs[1].set_ylabel('model prediction')

fig.tight_layout()
plt.show()
