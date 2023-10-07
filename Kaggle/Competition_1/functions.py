import pandas as pd
import matplotlib.pyplot as plt
import math
from xgboost import XGBClassifier

drop_cols=["id", "hospital_number", "lesion_2", "lesion_3", 'lesion_1']

def downloadData(path : str) -> pd.DataFrame:
    return pd.read_csv(path)


#We perform key transformations and predictor creation in this function
def transformData(data : pd.DataFrame) -> pd.DataFrame:
    def parseLesion(row, num):
        if row > 0:
            if len(str(row)): return ('0'+str(row))[num]
            return str(row)[num]
    #Create new instance of data frame so we can reference original dataframe later
    transformed_data = data.copy() 

    lesion_list = transformed_data['lesion_1']

    for idx, n in enumerate(lesion_list):
        n_str = str(n)
        if len(n_str) > 4:
            if int(n_str[0:2]) > 11:
                transformed_data.at[idx, 'lesion_1_site'] = n_str[0]
                transformed_data.at[idx, 'lesion_1_type'] = n_str[1]
                transformed_data.at[idx, 'lesion_1_subtype'] = n_str[2]
                transformed_data.at[idx, 'lesion_1_specific_code'] = n_str[3:]
            else:
                transformed_data.at[idx, 'lesion_1_site'] = n_str[0:2]
                transformed_data.at[idx, 'lesion_1_type'] = n_str[2]
                transformed_data.at[idx, 'lesion_1_subtype'] = n_str[3]
                transformed_data.at[idx, 'lesion_1_specific_code'] = n_str[4]
        else:
            transformed_data.at[idx, 'lesion_1_site'] = str(int(n // 1000))
            transformed_data.at[idx, 'lesion_1_type'] = str(int((n % 1000) // 100))
            transformed_data.at[idx, 'lesion_1_subtype'] = str(int((n % 100) // 10))
            transformed_data.at[idx, 'lesion_1_specific_code'] = str(int(n % 10))
    
    transformed_data['lesion_2_ind'] = transformed_data['lesion_2'].apply(lambda x: 1 if x > 0 else 0)   

    #Drop cols not needed
    for col in drop_cols:
        transformed_data.drop(col, axis=1, inplace=True)


    #Create dummy variables here
    transformed_data = createDummyVariables(transformed_data)

    return transformed_data

#Plots a simple frequency histogram
def frequencyHistogram(colname : str, data : pd.DataFrame):

    counts = data[colname].value_counts()
    proportion = counts/counts.sum()

    #Create histogram plot here
    plt.figure()
    plot = proportion.plot(kind = 'bar', alpha = 0.6, color='green', edgecolor = 'black')
    plot.grid(alpha = 0.3)
    plot.set_xlabel(f'{colname}', fontweight='bold')
    plot.set_ylabel('Frequency', fontweight='bold')
    plot.set_title(f'Overall {colname} Distribution')
    plt.show()
    


def imputeCols(data: pd.DataFrame) -> pd.DataFrame:

        for col in data.columns:
            if isColumnCategorical(data[col], col): categorical_cols.append(col)


        return data
#Iterates through each column and creates dummies for categorical columns.
def createDummyVariables(data: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = []

    for col in data.columns:
        if isColumnCategorical(data[col], col): categorical_cols.append(col)
    
    return pd.get_dummies(data=data, columns=categorical_cols, drop_first = True)


#Simply finds out if a column is categorical
def isColumnCategorical(data : pd.Series, colname : str) -> bool:

    if len(data.value_counts()) > 12: return False
    return True

#Create our response variable.
def createResponse(response : pd.DataFrame) -> pd.DataFrame:
    def applyResponseMap(val : str) -> int:
        if val == 'died': return 0
        elif val == 'euthanized': return 1
        else: return 2

    return response.apply(lambda row: applyResponseMap(row))

#Uncreate our response variable for submission.
def uncreateResponse(responses : list[int], test_data : pd.DataFrame) -> list[str]:
    prediction = []

    def applyResponseMap(val : int) -> int:
        if val == 0: return 'died'
        elif val == 1: return 'euthanized'
        else: return 'lived'


    for response in responses:
        prediction.append(applyResponseMap(response))

    prediction = pd.DataFrame({'outcome' : prediction})
    prediction['id'] = test_data['id']
    return prediction


'''
Finds columns that are prsent in training data but not intest data and adds blank column so model can predict
Finds columns that are in test data but not in training data and drops them.

Essentially keeps all dummy variables consistent with training data.
'''

def repairTestData(train_data : pd.DataFrame, test_data : pd.DataFrame, model : XGBClassifier) -> pd.DataFrame:

    train_cols = train_data.columns
    test_cols = test_data.columns

    for col in test_cols:
        if col not in train_cols: test_data.drop(col, axis=1, inplace=True)
    
    for col in train_cols:
        if col not in test_cols: test_data[col] = 0

    test_data = test_data[model.get_booster().feature_names]

    return test_data
