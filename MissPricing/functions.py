import pandas as pd
import os
import openai as ai
import datetime
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torch import nn
import re
import numpy as np
import copy
from sklearn.linear_model import GammaRegressor

class simulateRisk:

    __data = None
    __expected = None

    def __init__(self, low : bool, size : int):

        x_1 = np.random.normal(loc = 5, scale = 1, size = size)
        x_2 = np.random.normal(loc = 5, scale = 1, size = size)
        x_3 = 0 if low else 1

        def __simulateTheta(low : bool, size : int):

            # we might need to vary the betas by risk class. For now they are equal.
            b_1 = np.random.normal(loc = 5, scale = 0.5, size = size)
            b_2 = np.random.normal(loc = 5, scale = 0.5, size = size)
            b_3 = np.random.normal(loc = 5, scale = 0.5, size = size)

            if low: theta = 1 + b_1*x_1 + b_2*x_2 
            else: theta = 2 + b_1*x_1 + b_2*x_2 + b_3*x_3

            return theta
        
        def __simulateAlpha(low : bool, size : int):

            # we might need to vary the betas by risk class. For now they are equal.
            b_1 = np.random.normal(loc = 5, scale = 0.5, size = size)

            if low: alpha = 1 + 0.1*b_1*x_1
            else: alpha = 2 + 0.2*b_1*x_1

            return alpha


        theta = __simulateTheta(low = low, size = size)
        alpha = __simulateAlpha(low = low, size = size)

        gamma_samples = np.random.gamma(shape = alpha, scale = theta, size = size)

        self.data = pd.DataFrame({'Response' : gamma_samples, 'x1' : x_1, 'x2' : x_2, 'x3' : x_3})
        self.expected = pd.DataFrame({'True Expected' : alpha*theta})
    
    def getData(self) -> pd.DataFrame:
        return self.data
   

    def getExpected(self) -> pd.DataFrame:
        return self.expected    

class SimulatePortfolio:

    __highmeanpct = None
    __lowmeanpct = None
    __low = None
    __high = None
    

    def __init__(self, iterations : int, profit : float, size : int):

        self.__profit = profit
        self.__iterations = iterations
        self.__size = size


    def runSimulation(self):

        differences_low_mean_list = []
        differences_high_mean_list = []

        differences_low_list = []
        differences_high_list = []

        for i in range(self.__iterations):
            if i%50==49: print(f'100 iterations complete, {sel.__iterations-i} remaining.')
            low_risk = simulateRisk(True, self.__size)
            high_risk = simulateRisk(False, self.__size)

            low_risk_samples = low_risk.getData()
            high_risk_samples = high_risk.getData()

            low_risk_expected = low_risk.getExpected()
            high_risk_expected = high_risk.getExpected()

            #Seperate samples from known risk characteristics
            Y_low = low_risk_samples['Response'].to_frame()
            X_low = low_risk_samples.drop('Response', axis=1)

            Y_high = high_risk_samples['Response'].to_frame()
            X_high = high_risk_samples.drop('Response', axis=1)

            Y_total = pd.concat([Y_low, Y_high])
            X_total = pd.concat([X_low, X_high])
            total_expected = pd.concat([low_risk_expected, high_risk_expected])

            model = GammaRegressor(solver = 'newton-cholesky')
            model.fit(X = X_total, y = Y_total)

            #Find the differences between predicted and true expected losses
            predicted_losses_total = model.predict(X_total)
            predicted_losses_low = model.predict(X_low)
            predicted_losses_high = model.predict(X_high)

            differences_total = abs(total_expected.to_numpy().ravel() - predicted_losses_total)
            differences_low = abs(low_risk_expected.to_numpy().ravel() - predicted_losses_low)
            differences_high = abs(high_risk_expected.to_numpy().ravel() - predicted_losses_high)

            differences_low_mean = np.sum(differences_low)/np.sum(predicted_losses_low)
            differences_high_mean = np.sum(differences_high)/np.sum(predicted_losses_high)

            differences_low_mean_list.append(differences_low_mean)
            differences_high_mean_list.append(differences_high_mean)

            differences_high_list.append(differences_high)
            differences_low_list.append(differences_low)
            
        self.__highmeanpct = differences_high_mean_list
        self.__lowmeanpct = differences_low_mean_list

        self.__low = differences_low_list
        self.__high = differences_high_list

    def getHighMeanPercent(self):
        return self.__highmeanpct
    
    def getLowMeanPercent(self):
        return self.__lowmeanpct
    
    def getLowDifferences(self):
        return self.__low
    
    def getHighDifferences(self):
        return self.__high

def sampleDiagnositcs(samples : pd.DataFrame) -> None:

    samples = samples.to_numpy()

    mean = np.mean(samples)
    std = np.std(samples)
    min = np.min(samples)
    max = np.max(samples)

    q1 = np.quantile(samples, 0.01)
    q25 = np.quantile(samples, 0.25)
    q75 = np.quantile(samples, 0.25)
    q99 = np.quantile(samples, 0.99)

    print("Here are the requested diagnostics:")
    print(f"mean: {mean:.2f}")
    print(f"std: {std:.2f}")
    print(f"min: {min:.2f}")    
    print(f"q1%: {q1:.2f}")
    print(f"q25%: {q25:.2f}")
    print(f"q75%: {q75:.2f}")
    print(f"q99%: {q99:.2f}")
    print(f"max: {max:.2f}")    