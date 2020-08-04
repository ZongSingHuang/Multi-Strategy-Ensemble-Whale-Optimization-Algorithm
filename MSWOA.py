# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:59:58 2020

@author: ZongSing_NB

Main reference:http://www.alimirjalili.com/WOA.html
"""

import math
import numpy as np
import matplotlib.pyplot as plt

class MSWOA():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500,
                 b=1, x_max=1, x_min=0, a_max=2, a_min=0, l_max=1, l_min=-1, a2_max=-1, a2_min=-2):
        self.fit_func = fit_func        
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter     
        self.x_max = x_max
        self.x_min = x_min
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b

        self._iter = 1
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)
        
        self.X = np.random.uniform(low=-1.0, high=1.0, size=[self.num_particle, self.num_dim])
        self.X = 1 - 2*( np.cos( 4*np.arccos(self.X) ) )**2
        self.X = (self.X+1) / 2
        self.X = self.X*(self.x_max-self.x_min) + self.x_min
        
        score = self.fit_func(self.X)
        self.gBest_score = score.min().copy()
        self.gBest_X = self.X[score.argmin()].copy()
        self.gBest_curve[0] = self.gBest_score.copy()
        
    def opt(self):
        while(self._iter<self.max_iter):
            a = self.a_max - (self.a_max-self.a_min)*(self._iter/self.max_iter)
            # a = 2*np.cos(self._iter/self.max_iter)
            
            for i in range(self.num_particle):
                p = np.random.uniform()
                R1 = np.random.uniform()
                R2 = np.random.uniform()
                R3 = np.random.uniform()
                R4 = np.random.uniform()
                R5 = np.random.uniform()
                R6 = np.random.uniform()
                A = 2*a*R1 - a
                C = 2*R2
                l = np.random.uniform()*(self.l_max-self.l_min) + self.l_min

                if p<0.5:
                    if np.abs(A)<1:
                        D = np.abs(C*self.gBest_X - self.X[i, :])
                        self.X[i, :] = self.gBest_X - A*D
                    else:
                        self.X[i, :] = self.X[i, :]*R3 + 2*R4*np.abs(self.gBest_X - self.X[i, :])
                else:                    
                    D = np.abs(self.gBest_X - self.X[i, :])
                    if np.abs(A)<1:
                        self.X[i, :] = self.gBest_X + D*np.exp(self.b*l)*np.cos(2*np.pi*l)*self.levy(size=30)
                    else:
                        self.X[i, :] = self.X[i, :] + D*np.exp(self.b*l)*np.cos(2*np.pi*l)*self.levy(size=30)                
                
            bound_max = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_max[np.newaxis, :])
            bound_min = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_min[np.newaxis, :])
            idx_too_high = bound_max < self.X
            idx_too_low = bound_min > self.X
            bound_max_map = bound_max[idx_too_high] + \
                            R5*bound_max[idx_too_high]*(bound_max[idx_too_high]-self.X[idx_too_high])/self.X[idx_too_high]
            bound_min_map = bound_min[idx_too_low] + \
                            R6*np.abs(bound_min[idx_too_low]*(bound_min[idx_too_low]-self.X[idx_too_low])/self.X[idx_too_low])
            if np.any(bound_max_map==np.inf) or np.any(bound_min_map==np.inf):                   
                print(123)                
            self.X[idx_too_high] = bound_max_map.copy()
            self.X[idx_too_low] = bound_min_map.copy()
            score = self.fit_func(self.X)
            if np.min(score) < self.gBest_score:
                self.gBest_X = self.X[score.argmin()].copy()
                self.gBest_score = score.min().copy()
                
            self.gBest_curve[self._iter] = self.gBest_score.copy()    
            self._iter = self._iter + 1
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()  
        
    def levy(self, size=1):
        beta = 1.5
        
        sigma_u_up = math.gamma(1+beta) * np.sin(np.pi*beta/2)
        sigma_u_down = math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)
        sigma_u = (sigma_u_up / sigma_u_down)**(1/beta)
        
        u = np.random.normal(0, sigma_u**2, size=size)
        v = np.random.normal(0, 1, size=size)
        s = u / ( np.abs(v)**(1/beta) )
        
        return s
            