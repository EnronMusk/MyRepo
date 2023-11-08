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

ai.api_key = os.environ.get('AI_API_KEY')


key_words = [
    "excellent",
    "good",
    "great",
    "impressive",
    "satisfactory",
    "outstanding",
    "fantastic",
    "awesome",
    "wonderful",
    "superb",
    "positive",
    "commendable",
    "satisfying",
    "pleasing",
    "exceptional",
    "positive",
    "terrific",
    "amazing",
    "marvelous",
    "splendid",
    "phenomenal",
    "top-notch",
    "exemplary",
    "admirable",
    "praiseworthy",
    "stellar",
    "splendiferous",
    "sublime",
    "pristine",
    "magnificent",
    "flawless",
    "perfect",
    "poor",
    "bad",
    "subpar",
    "inferior",
    "unsatisfactory",
    "abysmal",
    "lousy",
    "mediocre",
    "atrocious",
    "dreadful",
    "inferior",
    "awful",
    "terrible",
    "horrendous",
    "dismal",
    "displeased",
    "dissatisfied",
    "defective",
    "faulty",
    "malfunctioning",
    "improper",
    "inadequate",
    "expensive",
    "costly",
    "pricey",
    "overpriced",
    "budget-friendly",
    "affordable",
    "reasonably priced",
    "economical",
    "value for money",
    "high-priced",
    "low-priced",
    "performance",
    "speed",
    "laggy",
    "responsive",
    "smooth",
    "efficient",
    "powerful",
    "weak",
    "slow",
    "user-friendly",
    "easy to use",
    "intuitive",
    "challenging",
    "user-friendliness",
    "durability",
    "reliable",
    "dependable",
    "trustworthy",
    "high-quality",
    "premium",
    "substandard",
    "low-quality",
    "inferior",
    "brand",
    "manufacturer",
    "maker",
    "company",
    "firm",
    "producer",
    "bug",
    "glitch",
    "issue",
    "problem",
    "complication",
    "difficulty",
    "incompatibility",
    "accessory",
    "addon",
    "attachment",
    "extension",
    "component",
    "user experience",
    "interface",
    "interface design",
    "usability",
    "user interface",
    "user interface design",
    "compatibility",
    "compatibility issue",
    "update",
    "upgrade",
    "update problem",
    "context",
    "setting",
    "environment",
    "location",
    "place",
    "circumstances",
    "situation",
    "surroundings",
    "scenario",
    "circumstance",
    "impressed",
    "content",
    "delighted",
    "happy",
    "satisfied",
    "pleased",
    "joyful",
    "thrilled",
    "satisfied",
    "recommend",
    "love",
    "like",
    "admire",
    "enjoy",
    "convenient",
    "efficient",
    "effective",
    "reliable",
    "smooth",
    "flawless",
    "seamless",
    "innovative",
    "advanced",
    "highly",
    "top",
    "impeccable",
    "best",
    "pros",
    "advantage",
    "benefit",
    "ideal",
    "exceptional",
    "superior",
    "remarkable",
    "stellar",
    "perfectly",
    "brilliant",
    "splendid",
    "outstanding",
    "immaculate",
    "exquisite",
    "majestic",
    "fantastic",
    "phenomenal",
    "remarkable",
    "improvement",
    "upgrade",
    "amazing",
    "wow",
    "improve",
    "happy",
    "satisfied",
    "pleased",
    "impressed",
    "excited",
    "enthusiastic",
    "content",
    "grateful",
    "positive",
    "favorable",
    "commendable",
    "exceed",
    "improvement",
    "better",
    "perfect",
    "like-new",
    "fantastic",
    "wonderful",
    "exceptional",
    "superb",
    "flawless",
    "improved",
    "extraordinary",
    "top-notch",
    "impressive",
    "positive",
    "pleasing",
    "impressive",
    "success",
    "bravo",
    "excellence",
    "incredible",
    "quality",
    "awesome",
    "favorable",
    "outstanding",
    "satisfactory",
    "efficient",
    "durable",
    "recommended",
    "pleasing",
    "improvement",
    "advantageous",
    "value",
    "pleasant",
    "joyful",
    "ideal",
    "terrific",
    "succeed",
    "outstanding",
    "exceeds",
    "meet",
    "improve",
    "goodness",
    "superiority",
    "superiority",
    "phenomenal",
    "amazing",
    "excellent",
    "quality",
    "reliable",
    "impressed",
    "satisfied",
    "happy",
    "pleased",
    "effective",
    "efficiency",
    "smoothness",
    "convenience",
    "innovation",
    "advanced",
    "superior",
    "exceptional",
    "fantastic",
    "perfectly",
    "impeccable",
    "best",
    "brilliant",
    "splendid",
    "outstanding",
    "immaculate",
    "exquisite",
    "majestic",
    "superb",
    "phenomenal",
    "stellar",
    "awesome",
    "wonderful",
    "improvement",
    "upgrade",
    "amazing",
    "exceed",
    "improve",
    "fantastic",
    "wow",
    "improve",
    "highly recommend",
    "top choice",
    "great value",
    "top-notch",
    "highly impressed",
    "improved my life",
    "couldn't be happier",
    "a game-changer",
    "game-changing",
    "life-changing",
    "truly exceptional",
    "far exceeded my expectations",
    "flawless experience",
    "money well spent",
    "can't live without it",
    "couldn't ask for more",
    "top of the line",
    "worth every penny",
    "must-have",
    "incredible value",
    "impressed beyond words",
    "outstanding performance",
    "beyond amazing",
    "exceeded all my hopes",
    "excellent investment",
    "couldn't be more satisfied",
    "changed my life",
    "ecstatic",
    "overjoyed",
    "thriving",
    "wonderful",
    "awe-inspiring",
    "delightful",
    "jubilant",
    "blissful",
    "superlative",
    "aces",
    "champion",
    "grand",
    "miraculous",
    "exhilarating",
    "jubilation",
    "heartwarming",
    "exultant",
    "radiant",
    "sizzling",
    "impressive",
    "appalling",
    "disastrous",
    "gruesome",
    "frustrating",
    "agonizing",
    "pitiful",
    "desperate",
    "inferiority",
    "troublesome",
    "detestable",
    "abominable",
    "repugnant",
    "lamentable",
    "abysmal",
    "revolting",
    "lousy",
    "displeasing",
    "dismaying",
    "termination",
    "apprehensive"
]

drop_cols=["reviewerID", "asin", "reviewerName", "helpful", "reviewText", "summary", "unixReviewTime", "reviewTime"]

def downloadData(path : str, n : int) -> pd.DataFrame:
    return pd.read_json(path, lines=True).head(n)

#Checks if a given word is in the review title or review body, returns 1 if true 0 else.
def wordCheck(row : pd.DataFrame, word : str) -> bool:
    return 1 if (word in row['reviewText'].lower() or word in row['summary'].lower()) else 0

#We perform key transformations and predictor creation in this function
def transformData(data : pd.DataFrame) -> pd.DataFrame:

    #Create new instance of data frame so we can reference original dataframe later
    transformed_data = data.copy() 

    transformed_data['helpful_ratio'] = transformed_data['helpful'].apply(lambda x: round(x[0] / (x[1]+1), 2))

    #Binary response variable
    transformed_data['overall_positive'] = transformed_data['overall'].apply(lambda row: 1 if row >= 3 else 0)

    #Center score variable
    transformed_data['overall'] = transformed_data['overall'].apply(lambda x: x-1)


    #Create keyword predictors
    for word in key_words:
        transformed_data[word] = transformed_data.apply(lambda row: wordCheck(row, word), axis=1)

    #Drop cols not needed
    for col in drop_cols:
        transformed_data.drop(col, axis=1, inplace=True)

    return transformed_data

'''
This function fetches GPT predictions.
We create a prompt with reviewText inserted.

'''
def fetchGPT(data : pd.DataFrame, collection : pd.DataFrame, Y_gbm : pd.DataFrame) -> pd.DataFrame:
    saveCSV(data, 'X_test_binary_gpt') #Save before erasing test dataset.

    #create new instance of data to return.
    data_return = data.copy()
    i = 0

    for index, row in data.iterrows():
        #Make sure we obey API limits
        time = datetime.datetime.now()

        #We use a counter to give progress reports, and return early to avoid timeout.
        if i == 25: 
            print(f"Successfully called 25 prompts. @ {time} Remaining prompts: {len(data_return)}")
            return data_return, collection
        i+=1

        review = row['reviewText']
        title = row['summary']


        
        prompt = f'Here is a product review, what score do you think the person who wrote the review gave the product on a scale of 1 to 5 with 1 being very negative and 5 being very positive. Format your answer as ONLY one character. No words should be present in your output. """{review}""" Also here is the title of the review """{title}"""'
        
        try: response = ai.ChatCompletion.create(model = "gpt-3.5-turbo", messages=[{'role': 'user', 'content': prompt}], temperature = 1, max_tokens=1000)
        except ai.error.RateLimitError: print("API limit reached, please try again in one minute.")


        prediction = int(response.choices[0]['message']['content'])-1
        prediction_binary = 1 if prediction >= 2 else 0


        df = pd.DataFrame([{'prediction': prediction, 'prediction_binary': prediction_binary}])
        collection = pd.concat(objs=[collection, df], ignore_index = True, axis=0)
        data_return.drop(index, axis=0, inplace=True)

    #Save datasets immediately after result ready.
    saveCSV(collection, 'predictions_binary_gpt_df')
    saveCSV(Y_gbm, 'Y_test_binary')
    print(f"Done. {len(collection)} predictions ready.")

    return data_return, collection

#Saves csv to results file
def saveCSV(data : pd.DataFrame, name : str):

    path = r'C:/Users/Luke/MyRepo/ReviewGPT/Results/'+name+r'.csv'
    data.to_csv(path)

#Data transform for the advanced model
def advTransformData(data : pd.DataFrame) -> pd.DataFrame:

    #Create new instance of data frame so we can reference original dataframe later
    transformed_data = data.copy() 

    #Binary response variable
    transformed_data['overall_positive'] = transformed_data['overall'].apply(lambda row: 1 if row >= 3 else 0)

    #Center score variable
    transformed_data['overall'] = transformed_data['overall'].apply(lambda x: x-1)

    word_list = createWordList(transformed_data)

    #Create keyword predictors
    for word in word_list:
        print(f"create columns {word}")
        transformed_data[word] = transformed_data.apply(lambda row: wordCheck(row, word), axis=1)
        if len(transformed_data.columns) >= 2000: break

    #Drop cols not needed
    for col in drop_cols:
        transformed_data.drop(col, axis=1, inplace=True)

    

    return transformed_data


#Creates a list of all possible words in the reviews (includes all)
def createWordList(data : pd.DataFrame) -> list[str]:
    adv_key_words = Counter()

    for index, row in data.iterrows():

        
        words_raw = re.sub(r'[^a-zA-Z ]', '', row['reviewText'])
        words_raw = words_raw.lower()
        words = words_raw.split()

        #Searches for words in the review, if they are not in dict then add else iterate counter.
        for word in words:
            if word not in ['how', 'who', 'what', 'when', 'where', 'why', 'is', 'the', 'a', 'an', 'and', 'to', 'with', 'in', 'so', 'overall', 'overall_positive']: adv_key_words[word] +=1

    return dropRareWords(adv_key_words, 50)

#This function drops infrequency words from the dictionary
def dropRareWords(dict : dict[str, int], n : int) -> list[str]:
    dict_mod = dict.copy()

    for key in dict.keys():
        if dict.get(key) < n: dict_mod.pop(key)

    return dict_mod.keys()

#PyTorch implementation of a nn
class NeuralNetworkBinary(nn.Module):
    def __init__(self, n_input : int, n_hidden_layer : int, n_output : int, learning_rate : float):
        super().__init__()

        #Setup the inputs and hidden layer and output
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_hidden_layer, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_hidden_layer, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_hidden_layer, n_hidden_layer),
            nn.ReLU(),
            nn.Linear(n_hidden_layer, n_output),
            nn.Sigmoid()
        )
        self.model.cuda()
        self.loss_function = nn.BCELoss()
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

            train_loss = self.loss_function(pred_y, y_train)
            train_acc = (torch.round(pred_y) == y_train).float().mean()
            train_accs.append(float(train_acc))
            train_losses.append(float(train_loss))

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
        
            #Review performance on test dataset. (no training here)
            self.model.eval() #Fix the optimzier and training loss.

            pred_y = self.model(x_test)
            pred_y = torch.squeeze(pred_y, 1)

            test_loss = self.loss_function(pred_y, y_test)
            test_acc = (torch.round(pred_y) == y_test).float().mean()
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

class ScoreNeuralNetwork(nn.Module):
    def __init__(self, n_input : int, n_hidden_layer : int, n_output : int, learning_rate : float, weights : torch.Tensor):
        super().__init__()

        #Setup the inputs and hidden layer and output
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden_layer, bias = False),
            nn.ReLU(),
            nn.Linear(n_input, n_hidden_layer, bias = False),
            nn.ReLU(),
            nn.Linear(n_input, n_hidden_layer, bias = False),
            nn.ReLU(),
            nn.Linear(n_hidden_layer, n_output, bias = False),
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

        best_acc = 0
        best_iter = 0
        best_model = None

        for i in range(batch_size):
            self.model.train() #Put model in training mode

            pred_y = self.model(x_train)
            pred_y = torch.squeeze(pred_y, 1)

            train_loss = self.loss_function(pred_y, y_train)
            train_acc = (torch.argmax(pred_y, 1) == torch.argmax(y_train, 1)).float().mean()
            train_accs.append(float(train_acc))
            train_losses.append(float(train_loss))

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
        
            #Review performance on test dataset. (no training here)
            self.model.eval() #Fix the optimzier and training loss.

            pred_y = self.model(x_test)
            pred_y = torch.squeeze(pred_y, 1)

            test_loss = self.loss_function(pred_y, y_test)
            test_acc = (torch.argmax(pred_y, 1) == torch.argmax(y_test, 1)).float().mean()
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

#Here we create a custom, ordinal encoder. This forces the NN to train on the classes as a oridinal set.
def ordinalEncoder(response :  pd.DataFrame, inverse = False) -> pd.DataFrame:

    response = response.values.tolist()
    encoder_mapping = {0 : [0,0,0,0], 1 : [1,0,0,0], 2 : [1,1,0,0], 3 : [1,1,1,0], 4 : [1,1,1,1]}
    encoded_response = []

    #Perform inverse mapping for predictions
    if inverse:
        inv_encoder_mapping = {str(val): key for key, val in encoder_mapping.items()} #Use string as dictionary keys cant be lists

        for row in response:
            encoded_response.append(inv_encoder_mapping[str(row)])
    else:
        encoded_response = []

        for row in response:
            encoded_response.append(encoder_mapping[row])

    return pd.DataFrame(encoded_response)

def inverseDummies(response : pd.DataFrame) -> pd.DataFrame:

    response = response.values.tolist()
    encoder_mapping = {0 : [1,0,0,0,0], 1 : [0,1,0,0,0], 2 : [0,0,1,0,0], 3 : [0,0,0,1,0], 4 : [0,0,0,0,1]}
    inv_encoder_mapping = {str(val): key for key, val in encoder_mapping.items()}
    encoded_response = []

    for row in response:
        encoded_response.append(inv_encoder_mapping[str(row)])

    return pd.DataFrame(encoded_response)

class ScoreNeuralNetworkOrdinal(nn.Module):
    def __init__(self, n_input : int, n_hidden_layer : int, n_output : int, learning_rate : float, weights : torch.Tensor):
        super().__init__()

        #Setup the inputs and hidden layer and output
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden_layer, bias = True),
            nn.ReLU(),
            nn.Linear(n_hidden_layer, n_output, bias = True),
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

        best_acc = 0
        best_iter = 0
        best_model = None

        for i in range(batch_size):
            self.model.train() #Put model in training mode

            pred_y = self.model(x_train)
            pred_y = torch.squeeze(pred_y, 1)

            train_loss = self.loss_function(pred_y, y_train)
            train_acc = (torch.argmax(torch.round(pred_y),1) == (torch.sum(y_train, dim=1))).float().mean()
            train_accs.append(float(train_acc))
            train_losses.append(float(train_loss))

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
        
            #Review performance on test dataset. (no training here)
            self.model.eval() #Fix the optimzier and training loss.

            pred_y = self.model(x_test)
            pred_y = torch.squeeze(pred_y, 1)

            test_loss = self.loss_function(pred_y, y_test)
            test_acc = (torch.argmax(torch.round(pred_y),1) == (torch.sum(y_test, dim=1))).float().mean()
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