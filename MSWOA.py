# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:59:58 2020

@author: ZongSing_NB

Main reference:
https://doi.org/10.3390/app10113667
"""

import math
import numpy as np
import matplotlib.pyplot as plt

class MSWOA():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0,
                 b=1, a_max=2, a_min=0, a2_max=-1, a2_min=-2, l_max=1, l_min=-1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub
        self.lb = lb
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b

        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
    def opt(self):
        # 初始化
        self.X = self.ChaoticInitializationStrategy()
        
        # 迭代
        for g in range(self.G):
            # 適應值計算
            F = self.fitness(self.X)
            
            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
                
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F

            # 更新
            a = self.a_max - (self.a_max-self.a_min)*(g/self.G) # (4)

            for i in range(self.P):
                p = np.random.uniform()
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                r3 = np.random.uniform()
                r4 = np.random.uniform()
                K = 2
                A = 2*a*(r1 - 1) # (2)
                C = 2*r2 # (3)
                l = np.random.uniform(low=self.l_min, high=self.l_max)
                D = np.abs(self.gbest_X - self.X[i, :])
                
                if p<0.5:
                    if np.abs(A)<1:
                        self.X[i, :] = self.gbest_X - A*np.abs(C*self.gbest_X - self.X[i, :]) # (1)
                    else:
                        self.X[i, :] = self.X[i, :]*r3 + K*r4*D # (9)
                else:
                    if np.abs(A)<1:
                        self.X[i, :] = self.gbest_X + D*np.exp(self.b*l)*np.cos(2*np.pi*l)*self.Levyflight() # (14)
                    else:
                        self.X[i, :] = self.X[i, :] + D*np.exp(self.b*l)*np.cos(2*np.pi*l)*self.Levyflight() # (13)
            
            self.ModifiedSpiralUpdatingPositionStrategy()
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.loss_curve[-1], 3))+']')
        plt.plot(self.loss_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()  
        
    def Levyflight(self):
        beta = 1.5
        f1 = math.gamma(1+beta)
        f2 = beta * math.gamma(1+beta) / 2
        f3 = np.sin(np.pi*beta/2)
        f4 = 2**( (beta-1)/2 )
        sigma_u = ( f1/f2 * f3/f4 ) ** (2/beta) # (12)
        sigma_v = 1.0 # (12)
        
        u = np.random.normal(0, sigma_u) # (11)
        v = np.random.normal(0, sigma_v) # (11)
        s = u / ( np.abs(v)**(1/beta) ) # (11)
        
        return s
    
    def ChaoticInitializationStrategy(self):
        m = 2
        n = 4
        X_scaled = np.zeros([self.P, self.D])
        X_scaled[0] = np.random.uniform(low=-1.0, high=1.0, size=[1, self.D])

        for i in range(1, self.P):
            X_scaled[i] = 1 - m*( np.cos( n*np.arccos(X_scaled[i-1]) ) )**2 # (8)
            
        X_std = ( X_scaled - (-1) ) / (1-(-1))
        X = X_std*(self.ub-self.lb) + self.lb
        
        return X
    
    def ModifiedSpiralUpdatingPositionStrategy(self):
        r5 = np.random.uniform()
        r6 = np.random.uniform()
        
        mask1 = self.X>self.ub
        mask2 = self.X<self.lb
        max_map = self.ub + r5*self.ub*(self.ub-self.X)/self.X # (15)
        min_map = self.lb + r6*np.abs(self.lb*(self.lb-self.X)/self.X) # (15)
        
        rand_X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        mask3 = max_map==np.inf
        mask4 = min_map==np.inf
        max_map[mask3] = rand_X[mask3]
        min_map[mask4] = rand_X[mask4]
        
        self.X[mask1] = max_map[mask1]
        self.X[mask2] = min_map[mask2]
        
        # 邊界處理後仍有超出值，因此需二次處理
        rand_X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        mask = np.logical_or(self.X>self.ub, self.X<self.lb)
        self.X[mask] = rand_X[mask].copy()