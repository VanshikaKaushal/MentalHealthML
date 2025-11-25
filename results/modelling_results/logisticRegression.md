=== Accuracy ===
0.8520114942528736

=== Classification Report ===
              precision    recall  f1-score   support

         0.0       0.83      0.81      0.82      2308
         1.0       0.87      0.89      0.88      3260

    accuracy                           0.85      5568
   macro avg       0.85      0.85      0.85      5568
weighted avg       0.85      0.85      0.85      5568


=== Confusion Matrix ===
[[1858  450]
 [ 374 2886]]

Misclassifies 450 class 0 → 1
Misclassifies 374 class 1 → 0
# Slight bias toward predicting class 1.

=== Feature Coefficients ===
                                 Feature  Coefficient
6   Have_you_ever_had_suicidal_thoughts_     2.492261
1                      Academic_Pressure     0.825664
8                       Financial_Stress     0.548558
17                   Degree_Category_PhD     0.333040
9       Family_History_of_Mental_Illness     0.259326
12             Degree_Category_STEM_Bach     0.174742
14                  Degree_Category_Arts     0.162732
16              Degree_Category_Graduate     0.151036
7                       Work_Study_Hours     0.114390
13              Degree_Category_Business     0.107403
2                                   CGPA     0.047941
15            Degree_Category_HighSchool     0.000343
0                                    Age    -0.118690
4                         Sleep_Duration    -0.142113
3                     Study_Satisfaction    -0.246579
5                         Dietary_Habits    -0.536214
11                         Gender_Female    -0.681650
10                           Gender_Male    -0.699168