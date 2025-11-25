=== Accuracy ===
0.8412356321839081

=== Classification Report ===
              precision    recall  f1-score   support

         0.0       0.82      0.79      0.80      2308
         1.0       0.85      0.88      0.87      3260

    accuracy                           0.84      5568
   macro avg       0.84      0.83      0.84      5568
weighted avg       0.84      0.84      0.84      5568


=== Confusion Matrix ===
[[1821  487]
 [ 397 2863]]

More mistakes compared to LR/XGB
Worst confusion matrix out of the three
# RF is clearly underperforming.

=== Feature Importances ===
                                 Feature  Importance
6   Have_you_ever_had_suicidal_thoughts_    0.223913
1                      Academic_Pressure    0.168524
8                       Financial_Stress    0.101806
0                                    Age    0.101429
7                       Work_Study_Hours    0.091230
2                                   CGPA    0.058984
3                     Study_Satisfaction    0.056792
4                         Sleep_Duration    0.046054
5                         Dietary_Habits    0.042420
9       Family_History_of_Mental_Illness    0.021014
12             Degree_Category_STEM_Bach    0.014286
10                           Gender_Male    0.013814
11                         Gender_Female    0.013774
16              Degree_Category_Graduate    0.013485
15            Degree_Category_HighSchool    0.010644
14                  Degree_Category_Arts    0.010091
13              Degree_Category_Business    0.008100
17                   Degree_Category_PhD    0.003639