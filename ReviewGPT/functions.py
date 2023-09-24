import pandas as pd
import numpy as np
import math as m
import matplotlib as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


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

"""
Checks if a word is present in review or review title. Outputs 1 for present 0 for not.

@params
word : the word to be checked.
row : the row to be searched for the word.
"""

def wordCheck(row : pd.DataFrame, word : str) -> bool:
    return 1 if (word in row['reviewText'].lower() or word in row['summary'].lower()) else 0

def transformData(data : pd.DataFrame) -> pd.DataFrame:
    data['helpful_ratio'] = data['helpful'].apply(lambda x: round(x[0] / (x[1]+1), 2))

    #Binary response variable
    data['overall_positive'] = data['overall'].apply(lambda row: 1 if row >= 3 else 0)


    #Create keyword predictors
    for word in key_words:
        data[word] = data.apply(lambda row: wordCheck(row, word), axis=1)

    #Drop cols not needed
    for col in drop_cols:
        data.drop(col, axis=1, inplace=True)

    return data

def frequencyHistogram(colname : str, data : pd.DataFrame):

    counts = data[colname].value_counts()
    proportion = counts/counts.sum()

    #Create histogram plot here
    plot = proportion.plot(kind = 'bar', alpha = 0.6, color='green', edgecolor = 'black')
    plot.grid(alpha = 0.3)
    plot.set_xlabel('Overall Score', fontweight='bold')
    plot.set_ylabel('Frequency', fontweight='bold')
    plot.set_title('Overall Score Distribution')
    
    
def returnKeywords() -> None:
    return key_words

#Prints simple diagnostics
def accuracy(Y_test : pd.DataFrame, predictions : list) -> None:
    accuracy = accuracy_score(Y_test, predictions)
    report = classification_report(Y_test, predictions)

    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')

#Creates and prints a simple confusion matrix
def confusionMatrix(Y_test : pd.DataFrame, predictions : list) -> None:
    tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()

    #Print the matrix
    print("Confusion Matrix:")
    print("")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")

#Prints importance scores of features
def importance(model : XGBClassifier) -> None:
    plot = xgb.plot_importance(model, max_num_features=15, importance_type='gain', values_format = '{v:.2f}')
    plot.grid(alpha=0.2)
    plot.set_xlabel('Gain', weight = 'bold')
    plot.set_ylabel('Features', weight = 'bold')