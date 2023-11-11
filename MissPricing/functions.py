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
import statsmodels.api as sm

class SimulateRisk:

    __data = None
    __expected = None

    def __init__(self, low : bool, size : int):

        x_0 = 1
        x_1 = np.random.normal(loc = 1, scale = 0.2, size = size)
        x_2 = np.random.normal(loc = 5, scale = 1, size = size)
        x_3 = 0 if low else 1

        x_1 = np.random.uniform(low = 0, high = 30, size = size)
        x_2 = np.random.uniform(low = 10, high = 20, size = size)

        # we might need to vary the betas by risk class. For now they are equal.
        b_1 = np.random.normal(loc = -0.1, scale = 0.01*0, size = size)
        b_2 = np.random.normal(loc = 0.03, scale = 0.05*0, size = size)
        b_3 = np.random.normal(loc = 0.05, scale = 0.3*0, size = size)

        b_h = np.random.gamma(shape = 5, scale = 20, size = size)

        def __simulateTheta(low : bool, size : int):

            theta = np.exp(-5*x_0 + (0.02)*x_1 + (0.1)*x_2) 

            return theta
        
        def __simulateAlpha(low : bool, size : int):

            alpha = np.exp(1*x_3)

            return alpha


        theta = __simulateTheta(low = low, size = size)
        alpha = __simulateAlpha(low = low, size = size)

        #For model training.
        gamma_samples = np.random.gamma(shape = alpha, scale = theta, size = size)

        #For independent samples for simulated profit metrics
        gamma_samples_extra = pd.DataFrame({'Sample' : np.random.gamma(shape = alpha, scale = theta, size = size)})

        self.data = pd.DataFrame({'Response' : gamma_samples, 'x0': x_0, 'x1' : x_1, 'x2' : x_2, 'x3' : x_3})
        self.expected = pd.DataFrame({'True Expected' : alpha*theta})
        self.extra = gamma_samples_extra
    
    def getData(self) -> pd.DataFrame:
        return self.data
   
    def getExpected(self) -> pd.DataFrame:
        return self.expected
    
    def getNewRandomSample(self) -> pd.DataFrame:
        return self.extra


class SimulatePortfolio:

    __highmeanpct = None
    __lowmeanpct = None
    __totalmeanpct = None
    __low = None
    __high = None
    __sigma_low = None
    __sigma_high = None
    lowAdj = None
    highAdj = None
    theoreticalProfit = None

    def __init__(self, iterations : int, profit : float, size : int):

        self.__profit = profit
        self.__iterations = iterations
        self.__size = size


    def runSimulation(self):

        differences_low_mean_list = []
        differences_high_mean_list = []
        differences_total_mean_list = []

        differences_low_list = []
        differences_high_list = []

        theoretical_profit_total_mean_list = []
        theoretical_profit_low_mean_list = []
        theoretical_profit_high_mean_list = []

        sampled_profit_total_mean_list = []
        sampled_profit_low_mean_list = []
        sampled_profit_high_mean_list = []

        for i in range(self.__iterations):
            if i%100==99: print(f'100 iterations complete, {self.__iterations-i-1} remaining.')
            low_risk = SimulateRisk(True, self.__size)
            high_risk = SimulateRisk(False, self.__size)

            low_risk_simulated = low_risk.getData()
            high_risk_simulated = high_risk.getData()

            low_risk_sampled = low_risk.getNewRandomSample()
            high_risk_sampled = high_risk.getNewRandomSample()
            total_risk_sampled = pd.concat([low_risk_sampled, high_risk_sampled])

            low_risk_expected = low_risk.getExpected()
            high_risk_expected = high_risk.getExpected()

            #Seperate simulated from known risk characteristics
            Y_low = low_risk_simulated['Response'].to_frame()
            X_low = low_risk_simulated.drop('Response', axis=1)

            Y_high = high_risk_simulated['Response'].to_frame()
            X_high = high_risk_simulated.drop('Response', axis=1)

            Y_total = pd.concat([Y_low, Y_high])
            X_total = pd.concat([X_low, X_high])
            total_expected = pd.concat([low_risk_expected, high_risk_expected])

            model = sm.GLM(Y_total, X_total, family = sm.families.Gamma(link=sm.families.links.Log())).fit()

            #Find the differences between predicted and true expected losses
            predicted_losses_total = model.predict(X_total)
            predicted_losses_low = model.predict(X_low)
            predicted_losses_high = model.predict(X_high)

            differences_total = abs(total_expected.to_numpy().ravel() - predicted_losses_total)
            differences_low = abs(low_risk_expected.to_numpy().ravel() - predicted_losses_low)
            differences_high = abs(high_risk_expected.to_numpy().ravel() - predicted_losses_high)

            #theoretical profit influenced by miss pricing
            theoretical_profit_total_mean = 1 - np.sum(total_expected.to_numpy().ravel())/np.sum(predicted_losses_total/(1-self.__profit))
            theoretical_profit_low_mean = 1 - np.sum(low_risk_expected.to_numpy().ravel())/np.sum((predicted_losses_low/(1-self.__profit)))
            theoretical_profit_high_mean = 1 - np.sum(high_risk_expected.to_numpy().ravel())/np.sum((predicted_losses_high/(1-self.__profit)))

            #Simulated profit metrics
            sampled_profit_total_mean = 1 - np.sum(total_risk_sampled.to_numpy().ravel())/np.sum(predicted_losses_total/(1-self.__profit))
            sampled_profit_low_mean = 1 - np.sum(low_risk_sampled.to_numpy().ravel())/np.sum((predicted_losses_low/(1-self.__profit)))
            sampled_profit_high_mean = 1 - np.sum(high_risk_sampled.to_numpy().ravel())/np.sum((predicted_losses_high/(1-self.__profit)))           

            #Calculate the mean miss pricing as the sum of the portfolio losses over the predicted losses for the portfolio
            differences_low_mean = np.sum(differences_low)/np.sum(predicted_losses_low.ravel())
            differences_high_mean = np.sum(differences_high)/np.sum(predicted_losses_high.ravel())
            differences_total_mean = np.sum(differences_total)/np.sum(predicted_losses_total.ravel())

            differences_low_mean_list.append(differences_low_mean)
            differences_high_mean_list.append(differences_high_mean)
            differences_total_mean_list.append(differences_total_mean)

            differences_high_list.append(differences_high)
            differences_low_list.append(differences_low)

            theoretical_profit_total_mean_list.append(theoretical_profit_total_mean)
            theoretical_profit_low_mean_list.append(theoretical_profit_low_mean)
            theoretical_profit_high_mean_list.append(theoretical_profit_high_mean)

            sampled_profit_total_mean_list.append(sampled_profit_total_mean)
            sampled_profit_low_mean_list.append(sampled_profit_low_mean)
            sampled_profit_high_mean_list.append(sampled_profit_high_mean)
            
        self.__highmeanpct = differences_high_mean_list
        self.__lowmeanpct = differences_low_mean_list
        self.__totalmeanpct = differences_total_mean_list

        self.__low = differences_low_list
        self.__high = differences_high_list

        #Calculate the risk adjusted profit margin
        self.lowAdj = np.mean(self.__lowmeanpct)/np.mean(self.__totalmeanpct)*self.__profit
        self.highAdj = np.mean(self.__highmeanpct)/np.mean(self.__totalmeanpct)*self.__profit

        #We need to make sure our risk adjusted profit margins are off-balanced to bring the portfolio yield up to standard.
        off_balance_factor = 1
        #self.__profit/((self.lowAdj + self.highAdj)/2)

        self.lowAdjOB = self.lowAdj*off_balance_factor
        self.highAdjOB = self.highAdj*off_balance_factor

        #Profit metrics
        self.theoreticalProfit = theoretical_profit_total_mean_list
        self.theoreticalLowProfit = theoretical_profit_low_mean_list
        self.theoreticalHighProfit = theoretical_profit_high_mean_list

        self.sampledProfit = sampled_profit_total_mean_list
        self.sampledLowProfit = sampled_profit_low_mean_list
        self.sampledHighProfit = sampled_profit_high_mean_list

    def runSimulationAdj(self):

        theoretical_profit_total_mean_list = []
        theoretical_profit_low_mean_list = []
        theoretical_profit_high_mean_list = []

        sampled_profit_total_mean_list = []
        sampled_profit_low_mean_list = []
        sampled_profit_high_mean_list = []

        for i in range(self.__iterations):
            if i%100==99: print(f'100 iterations complete, {self.__iterations-i-1} remaining.')
            low_risk = SimulateRisk(True, self.__size)
            high_risk = SimulateRisk(False, self.__size)

            low_risk_simulated = low_risk.getData()
            high_risk_simulated = high_risk.getData()

            low_risk_sampled = low_risk.getNewRandomSample()
            high_risk_sampled = high_risk.getNewRandomSample()

            low_risk_expected = low_risk.getExpected()
            high_risk_expected = high_risk.getExpected()

            #Seperate simulated from known risk characteristics
            Y_low = low_risk_simulated['Response'].to_frame()
            X_low = low_risk_simulated.drop('Response', axis=1)

            Y_high = high_risk_simulated['Response'].to_frame()
            X_high = high_risk_simulated.drop('Response', axis=1)

            Y_total = pd.concat([Y_low, Y_high])
            X_total = pd.concat([X_low, X_high])

            model = sm.GLM(Y_total, X_total, family = sm.families.Gamma(link=sm.families.links.Log())).fit()

            #Find the differences between predicted and true expected losses
            predicted_losses_low = model.predict(X_low)
            predicted_losses_high = model.predict(X_high)

            #theoretical,
            theoretical_profit_total_mean = 1 - np.sum(low_risk_expected.to_numpy().ravel() + high_risk_expected.to_numpy().ravel())/np.sum(predicted_losses_low/(1-self.lowAdjOB) + predicted_losses_high/(1-self.highAdjOB))
            theoretical_profit_low_mean = 1 - np.sum(low_risk_expected.to_numpy().ravel())/np.sum((predicted_losses_low/(1-self.lowAdjOB)))
            theoretical_profit_high_mean = 1 - np.sum(high_risk_expected.to_numpy().ravel())/np.sum((predicted_losses_high/(1-self.highAdjOB)))

            #Simulated profit metrics
            sampled_profit_total_mean = 1 - np.sum(low_risk_sampled.to_numpy().ravel() + high_risk_sampled.to_numpy().ravel())/np.sum(predicted_losses_low/(1-self.lowAdjOB) + predicted_losses_high/(1-self.highAdjOB))
            sampled_profit_low_mean = 1 - np.sum(low_risk_sampled.to_numpy().ravel())/np.sum((predicted_losses_low/(1-self.lowAdjOB)))
            sampled_profit_high_mean = 1 - np.sum(high_risk_sampled.to_numpy().ravel())/np.sum((predicted_losses_high/(1-self.highAdjOB)))

            theoretical_profit_total_mean_list.append(theoretical_profit_total_mean)
            theoretical_profit_low_mean_list.append(theoretical_profit_low_mean)
            theoretical_profit_high_mean_list.append(theoretical_profit_high_mean)

            sampled_profit_total_mean_list.append(sampled_profit_total_mean)
            sampled_profit_low_mean_list.append(sampled_profit_low_mean)
            sampled_profit_high_mean_list.append(sampled_profit_high_mean)


        #Calculate adjusted profit
        self.theoreticalProfitAdj = theoretical_profit_total_mean_list
        self.theoreticalLowProfitAdj = theoretical_profit_low_mean_list
        self.theoreticalHighProfitAdj = theoretical_profit_high_mean_list

        self.sampledProfitAdj = sampled_profit_total_mean_list
        self.sampledLowProfitAdj = sampled_profit_low_mean_list
        self.sampledHighProfitAdj = sampled_profit_high_mean_list


    def getHighMeanPercent(self):
        return self.__highmeanpct
    
    def getLowMeanPercent(self):
        return self.__lowmeanpct
    
    def getTotalMeanPercent(self):
        return self.__totalmeanpct
    
    def getLowDifferences(self):
        return self.__low
    
    def getHighDifferences(self):
        return self.__high
    
    def getSigmaLowList(self):
        return self.__sigma_low
    
    def getSigmaHighList(self):
        return self.__sigma_high
    
    def setProfitAdjHigh(self, profit : float):
        self.highAdjOB = profit
    
    def setProfitAdjLow(self, profit : float):
         self.lowAdjOB = profit       

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