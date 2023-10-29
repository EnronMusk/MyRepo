Here are the final results from the models:

Prompt used:
prompt = f'Here is a product review, what score do you think the person who wrote the review gave the product on a scale of 1 to 5 with 1 being very negative and 5 being very positive. Format your answer as ONLY one character. No words should be present in your output. """{review}""" Also here is the title of the review """{title}"""'

Promp tried with no difference in results:
prompt = f'Here is a product review, rate the review on a scale of 1 to 5 with 1 being very negative and 5 being very positive. Format your answer as ONLY one character. No words should be present in your output. """{review}""" Also here is the title of the review """{title}"""'

## Binary GBM
Accuracy: 0.8562
Classification Report:
              precision    recall  f1-score   support

           0       0.42      0.59      0.49      1180
           1       0.94      0.89      0.92      8820

    accuracy                           0.86     10000
   macro avg       0.68      0.74      0.70     10000
weighted avg       0.88      0.86      0.87     10000

## Binary ChatGPT
Accuracy: 0.947
Classification Report:
              precision    recall  f1-score   support

           0       0.72      0.69      0.70        91
           1       0.97      0.97      0.97       909

    accuracy                           0.95      1000
   macro avg       0.84      0.83      0.84      1000
weighted avg       0.95      0.95      0.95      1000

## Binary Neural Network
Accuracy: 0.8466360586261699
Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.84      0.85      5663
           1       0.84      0.85      0.85      5663

    accuracy                           0.85     11326
   macro avg       0.85      0.85      0.85     11326
weighted avg       0.85      0.85      0.85     11326

## Score GBM
Accuracy: 0.481
Classification Report:
              precision    recall  f1-score   support

           0       0.18      0.38      0.24        55
           1       0.03      0.06      0.04        36
           2       0.17      0.29      0.22        85
           3       0.34      0.37      0.35       207
           4       0.79      0.58      0.67       617

    accuracy                           0.48      1000
   macro avg       0.30      0.34      0.30      1000
weighted avg       0.58      0.48      0.52      1000

## Score ChatGPT
Accuracy: 0.569
Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.35      0.49        55
           1       0.27      0.47      0.35        36
           2       0.47      0.46      0.46        85
           3       0.33      0.71      0.45       207
           4       0.90      0.56      0.69       617

    accuracy                           0.57      1000
   macro avg       0.57      0.51      0.49      1000
weighted avg       0.72      0.57      0.60      1000

## Score Nerual Network (non ordinal)
Accuracy: 0.5232
Classification Report:
              precision    recall  f1-score   support

           0       0.10      0.06      0.08       653
           1       0.13      0.25      0.17       479
           2       0.00      0.00      0.00       798
           3       0.22      0.22      0.22      2041
           4       0.70      0.77      0.73      6029

    accuracy                           0.52     10000
   macro avg       0.23      0.26      0.24     10000
weighted avg       0.48      0.52      0.50     10000
