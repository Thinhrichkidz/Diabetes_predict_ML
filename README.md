# Diabetes_prediect
 
### Diabetes Prediction 
This repository contains code for predicting whether a patient has diabetes based on their medical history. The code uses a variety of machine learning techniques, including support vector machines, random forests, and lazy classification.

<h3>Getting Started</h3>
<p> To get started, clone this repository to your local machine. Then, install the dependencies using the following command: </p>

``` pip install -r requirements.txt ``` 

<h3> Usage </h3>
<p> To train the model, run the following command: </p>

``` python diabetes.py ``` 

<h3> Results </h3>

- The code was able to achieve an accuracy of 85% on the test set.
- The code was able to identify the most important features for predicting diabetes, which were age, sex, BMI, blood pressure, and glucose levels.
- The code was able to generate a confusion matrix, which showed that the model was better at predicting patients who did not have diabetes than patients who did have diabetes. 
- The code successfully trains and predicts the outcome of diabetes patients using a variety of machine learning models, including SVC, Random Forest, and LazyClassifier.
- The best model, as determined by GridSearchCV and RandomizedSearchCV, is a Random Forest with 1200 trees, max depth of 30, min samples split of 10, and min samples leaf of 2.

<h3> Limitatiosn </h3> 

- The code is not very generalizable to other datasets. The diabetes dataset is relatively small and well-behaved, so it is possible that the model would not perform as well on a larger or more complex dataset.
- The code does not take into account the uncertainty of the predictions. The accuracy of 85% means that there is a 15% chance that the model will make a wrong prediction. This uncertainty should be taken into account when making decisions based on the model's predictions.
