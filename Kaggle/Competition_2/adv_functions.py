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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def downloadData(path : str) -> pd.DataFrame:
    return pd.read_csv(path)


drop_cols =['id']

#We perform key transformations and predictor creation in this function
def transformData(data : pd.DataFrame) -> pd.DataFrame:

    #Create new instance of data frame so we can reference original dataframe later
    transformed_data = data.copy() 

    transformed_data = (transformed_data-transformed_data.mean())/transformed_data.std()

    #Drop cols not needed
    for col in drop_cols:
        transformed_data.drop(col, axis=1, inplace=True)

    return transformed_data

def classificationWeightCalculator(name : str, data : pd.DataFrame, device : str) -> torch.Tensor:
    weights = (data[name].value_counts().sort_index().iloc[-1]/data[name].value_counts().sort_index())

    print(weights)

    data = data.apply(lambda row: weights[row])
    data = torch.tensor(data.values, dtype = torch.float).to(device)
    return torch.squeeze(data, 1)

def classificationWeightByClass(name : str, data : pd.DataFrame, device : str) -> torch.Tensor:
    weights = (data[name].value_counts().sort_index().iloc[-1]/data[name].value_counts().sort_index())

    return torch.tensor(weights.values, dtype = torch.float).to(device)

#Pyrotch implementation of nn
class NeuralNetwork(nn.Module):
    def __init__(self, n_input : int, n_hidden_layer : int, n_output : int, learning_rate : float, weights_train : torch.Tensor, weights_test : torch.Tensor):
        super().__init__()

        #Setup the inputs and hidden layer and output
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden_layer, bias = False),
            nn.Linear(n_hidden_layer, n_hidden_layer, bias = False),
            nn.Linear(n_hidden_layer, n_hidden_layer, bias = False),
            nn.Linear(n_hidden_layer, n_output, bias = False),
            nn.Sigmoid()
        )
        self.model.cuda()
        self.loss_function_train = nn.NLLLoss(weight = weights_train)
        self.loss_function_test = nn.NLLLoss(weight = weights_test)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
    
    def train(self, x_train : torch.Tensor, y_train : torch.Tensor, batch_size : int, x_test : torch.Tensor, y_test : torch.Tensor) -> None:

        train_losses = []
        test_losses = []

        train_accs = []
        test_accs = []

        best_acc = 0
        best_iter = 0
        best_model = None

        for i in range(batch_size):
            self.model.train() #Put model in training mode

            pred_y = self.model(x_train)
            pred_y = torch.squeeze(pred_y, 1)

            train_loss = self.loss_function_train(pred_y, y_train)
            train_acc = (pred_y.round() == y_train).float().mean()
            train_accs.append(float(train_acc))
            train_losses.append(float(train_loss))

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
        
            #Review performance on test dataset. (no training here)
            self.model.eval() #Fix the optimzier and training loss.

            pred_y = self.model(x_test)
            pred_y = torch.squeeze(pred_y, 1)

            test_loss = self.loss_function_test(pred_y, y_test)
            test_acc = (pred_y.round() == y_test).float().mean()
            test_accs.append(test_acc.item())
            test_losses.append(test_loss.item())

            if i%250 == 249: print(f'250 iterations complete, current test accuracy: {test_acc:.2f}, curr entropy: {train_loss}')
            if test_acc > best_acc: 
                best_acc = test_acc
                best_iter = i
                best_model = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_model)
        #Plots training summary results on loss function
        plt.plot(train_losses, color='red', label='train')
        plt.plot(test_losses, color='black', label='test')
        plt.grid(alpha=0.3)
        plt.ylabel('loss',fontweight='bold')
        plt.xlabel('iteration',fontweight='bold')
        plt.title("Training and Test Loss Distribution")
        plt.legend()
        plt.show()

        #Plots training summary results on accuracy
        plt.plot(train_accs, color='red', label='train')
        plt.plot(test_accs, color='black', label='test')
        plt.grid(alpha=0.3)
        plt.ylabel('accuracy',fontweight='bold')
        plt.xlabel('iteration',fontweight='bold')
        plt.title("Training and Test Accuracy Distribution")
        plt.legend()
        plt.show()
        print(f'best accuracy achieved at iteration {best_iter} with accuracy {best_acc:.2f}')

#Finds the ideal weight for the ensemble model
def weightFinder(probs1, probs2, probs3, true_vals, weights) -> list[int]:
    n = len(probs1)
    max_auc = 0
    w1 = 0
    w2 = 0
    w3 = 0

    for i in range(1,20):
        for j in range(1,20):
            for k in range(1,20):
                a = min(1/i,0.9)
                b = min(1/j,0.9)
                c = min(1/k,0.9)
                probs_calc = a*probs1 + b*probs2 + c*probs3
                score = roc_auc_score(true_vals, probs_calc, sample_weight = weights)

                if score > max_auc and a+b+c<=1: 
                    max_auc = score
                    w1 = a
                    w2 = b
                    w3 = c

    print(f'max auc found {max_auc}')
    return [w1,w2,w3]

class ScoreNeuralNetwork(nn.Module):
    def __init__(self, n_input : int, n_hidden_layer : int, n_output : int, learning_rate : float, weights : torch.Tensor):
        super().__init__()

        #Setup the inputs and hidden layer and output
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_input, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_input, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_input, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_input, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_input, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_hidden_layer, n_output),
            nn.Sigmoid()
        )
        self.model.cuda()
        self.loss_function = nn.CrossEntropyLoss(weight = weights)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
    
    def train(self, x_train : torch.Tensor, y_train : torch.Tensor, batch_size : int, x_test : torch.Tensor, y_test : torch.Tensor) -> None:

        train_losses = []
        test_losses = []

        train_accs = []
        test_accs = []

        train_aucs = []
        test_aucs = []

        best_acc = 0
        best_iter = 0
        best_auc = 0
        best_model = None

        for i in range(batch_size):
            self.model.train() #Put model in training mode

            pred_y = self.model(x_train)
            pred_y = torch.squeeze(pred_y, 1)

            train_loss = self.loss_function(pred_y, y_train)
            train_acc = (torch.argmax(pred_y, 1) == torch.argmax(y_train, 1)).float().mean()
            train_auc = roc_auc_score(y_train[:,1].detach().cpu().numpy(), pred_y[:,1].detach().cpu().numpy())

            train_accs.append(float(train_acc))
            train_losses.append(float(train_loss))
            train_aucs.append(train_auc)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
        
            #Review performance on test dataset. (no training here)
            self.model.eval() #Fix the optimzier and training loss.

            pred_y = self.model(x_test)
            pred_y = torch.squeeze(pred_y, 1)

            test_loss = self.loss_function(pred_y, y_test)
            test_acc = (torch.argmax(pred_y, 1) == torch.argmax(y_test, 1)).float().mean()
            test_auc = roc_auc_score(y_test[:,1].detach().cpu().numpy(), pred_y[:,1].detach().cpu().numpy())

            test_accs.append(test_acc.item())
            test_losses.append(test_loss.item())
            test_aucs.append(test_auc)

            if i%250 == 249: print(f'250 iterations complete, current test accuracy: {test_acc:.2f}, curr entropy: {train_loss}, curr auc: {test_auc:.2f}')
            if test_acc > best_acc: 
                best_acc = test_acc
            if test_auc > best_auc:
                best_auc = test_auc
                best_iter = i
                best_model = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_model)
        #Plots training summary results on loss function
        plt.plot(train_losses, color='red', label='train')
        plt.plot(test_losses, color='black', label='test')
        plt.grid(alpha=0.3)
        plt.ylabel('loss',fontweight='bold')
        plt.xlabel('iteration',fontweight='bold')
        plt.title("Training and Test Loss Distribution")
        plt.legend()
        plt.show()

        #Plots training summary results on accuracy
        plt.plot(train_accs, color='red', label='train')
        plt.plot(test_accs, color='black', label='test')
        plt.grid(alpha=0.3)
        plt.ylabel('accuracy',fontweight='bold')
        plt.xlabel('iteration',fontweight='bold')
        plt.title("Training and Test Accuracy Distribution")
        plt.legend()
        plt.show()
        print(f'best accuracy achieved at iteration {best_iter} with accuracy {best_acc:.2f}')

        #Plots training summary results on auc
        plt.plot(train_aucs, color='red', label='train')
        plt.plot(test_aucs, color='black', label='test')
        plt.grid(alpha=0.3)
        plt.ylabel('auc',fontweight='bold')
        plt.xlabel('iteration',fontweight='bold')
        plt.title("Training and Test Auc Distribution")
        plt.legend()
        plt.show()
        print(f'best auc achieved at iteration {best_iter} with auc {best_auc:.2f}')

        #roc curve
        with torch.no_grad():
            pred_y = self.model(x_test)
            pred_y = torch.squeeze(pred_y, 1)

            pred_y = pred_y.numpy(force=True)
            y_test = y_test.numpy(force=True)
            pred_y = pred_y[:, 1]
            y_test = y_test[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test, pred_y)
            plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
            plt.title("Roc Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.show()



def inverseDummies(response : pd.DataFrame) -> pd.DataFrame:

    response = response.values.tolist()
    encoder_mapping = {0 : [1,0], 1 : [0,1]}
    inv_encoder_mapping = {str(val): key for key, val in encoder_mapping.items()}
    encoded_response = []

    for row in response:
        encoded_response.append(inv_encoder_mapping[str(row)])

    return pd.DataFrame(encoded_response)