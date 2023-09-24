import pandas as pd
import numpy as np
import matplotlib as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


#Plots a simple frequency histogram
def frequencyHistogram(colname : str, data : pd.DataFrame):

    counts = data[colname].value_counts()
    proportion = counts/counts.sum()

    #Create histogram plot here
    plot = proportion.plot(kind = 'bar', alpha = 0.6, color='green', edgecolor = 'black')
    plot.grid(alpha = 0.3)
    plot.set_xlabel('Overall Score', fontweight='bold')
    plot.set_ylabel('Frequency', fontweight='bold')
    plot.set_title('Overall Score Distribution')

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