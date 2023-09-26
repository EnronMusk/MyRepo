import pandas as pd
import os
import openai as ai
from ratelimit import limits, sleep_and_retry
import datetime

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
    saveCSV(data, 'X_test_binary_gpt')
    saveCSV(collection, 'predictions_binary_gpt_df')
    saveCSV(Y_gbm, 'Y_test_binary')
    print("Done. Predictions ready.")

    return data_return, collection

#Saves csv to results file
def saveCSV(data : pd.DataFrame, name : str):

    path = r'C:/Users/Luke/MyRepo/ReviewGPT/Results/'+name+r'.csv'
    data.to_csv(path)
