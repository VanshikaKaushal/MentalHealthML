=== Accuracy ===
0.852191091954023

=== Classification Report ===
              precision    recall  f1-score   support

         0.0       0.83      0.80      0.82      2308
         1.0       0.86      0.89      0.88      3260

    accuracy                           0.85      5568
   macro avg       0.85      0.85      0.85      5568
weighted avg       0.85      0.85      0.85      5568


=== Confusion Matrix ===
[[1854  454]
 [ 369 2891]]

Best recall for class 1 (only 369 mistakes)
Similar to logistic regression for class 0
# XGB gives the most stable performance.

=== Feature Importances ===
                                 Feature  Importance
6   Have_you_ever_had_suicidal_thoughts_    0.597707
1                      Academic_Pressure    0.121431
8                       Financial_Stress    0.065453
5                         Dietary_Habits    0.031297
0                                    Age    0.024683
7                       Work_Study_Hours    0.020466
3                     Study_Satisfaction    0.019449
4                         Sleep_Duration    0.014724
15            Degree_Category_HighSchool    0.014190
9       Family_History_of_Mental_Illness    0.012332
11                         Gender_Female    0.011273
2                                   CGPA    0.010179
10                           Gender_Male    0.009832
12             Degree_Category_STEM_Bach    0.009757
16              Degree_Category_Graduate    0.009747
13              Degree_Category_Business    0.009214
14                  Degree_Category_Arts    0.009159
17                   Degree_Category_PhD    0.009107