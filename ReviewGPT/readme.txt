Here are the final results from the models:

Binary GBM
Accuracy: 0.742
Classification Report:
              precision    recall  f1-score   support

           0       0.22      0.71      0.34        91
           1       0.96      0.74      0.84       909

    accuracy                           0.74      1000
   macro avg       0.59      0.73      0.59      1000
weighted avg       0.90      0.74      0.79      1000

Binary ChatGPT
Accuracy: 0.831
Classification Report:
              precision    recall  f1-score   support

           0       0.04      0.03      0.03        91
           1       0.90      0.91      0.91       909

    accuracy                           0.83      1000
   macro avg       0.47      0.47      0.47      1000
weighted avg       0.82      0.83      0.83      1000

Score GBM
Accuracy: 0.45
Classification Report:
              precision    recall  f1-score   support

           0       0.17      0.45      0.24        55
           1       0.15      0.28      0.19        36
           2       0.15      0.22      0.18        85
           3       0.28      0.31      0.30       207
           4       0.79      0.54      0.64       617

    accuracy                           0.45      1000
   macro avg       0.31      0.36      0.31      1000
weighted avg       0.57      0.45      0.49      1000

Score ChatGPT
Accuracy: 0.35
Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        55
           1       0.03      0.06      0.04        36
           2       0.08      0.08      0.08        85
           3       0.21      0.44      0.29       207
           4       0.63      0.40      0.49       617

    accuracy                           0.35      1000
   macro avg       0.19      0.20      0.18      1000
weighted avg       0.44      0.35      0.37      1000
