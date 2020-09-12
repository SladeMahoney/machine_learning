# Supervised Machine Learning

## Credit Risk Analysis 
This analysis utilizes several machine learning models and techniques to predict credit risk for a firm that wants to improve the way they manage loan applications and assess candidates for loans. The dataset used for this analysis is a csv file from LendingClub, a lending services company, which provides detailed information on the type of loans and applicants.

Before creating predictions, the data was prepared and split by the target variable "loan_status" and the other columns that help predict the target values. Identifying credit risk is an unbalanced classification problem. The ratio of good loans outweighs the risky loans, therefore a variety of sampling techniques were used to get a sample for training and evaluation. The performance of each machine learning model was evaluated through library tools that produce classification reports, confusion matrices, and accuracy scores.

Tools Used: Python, Jupyter Notebook, numpy, pandas, scikit-learn, imblearn
File: LoanStats2019Q1.csv

## Oversampling using Random and SMOTE
### Balanced Accuracy Scores
Random: .649
SMOTE: .64

### Confusion Matrices
Random: 
True Positive: 54 | False Positive: 33
False Negative: 5498 | True Negative: 11620

SMOTE:
True Positive: 55| False Positive: 32
False Negative: 5879| True Negative: 11239

### Precision and Recall Scores
Random: 
Precision (high-risk):.01  (low risk):1
Recall (high-risk): .62 (low-risk): .68

SMOTE: 
Precision (high-risk): .01 (low risk):1
Recall (high-risk): .63 (low-risk):.66

## Undersampling UsingC Cluster Centroids
### Balanced Accuracy Score
Centroids: .51

### Confusion Matrix: 
True Positive: 55| False Positive: 32
False Negative: 10346| True Negative:6772

### Precision and Recall Scores
Precision (high-risk): .01 (low risk):1
Recall (high-risk):.63 (low-risk):.40

## Combination (Over and Under) Sampling with SMOTENN
### Balanced Accuracy Score
SMOTENN: .62

### Confusion Matrix: 
True Positive: 59 | False Positive: 28
False Negative: 7347 | True Negative: 9771

### Precision and Recall Scores
Precision (high-risk): .01 (low risk):1
Recall (high-risk): .68 (low-risk):.57

### Summary
All models fit a perfect precision score for low risk detections and had solid F1 scores between .73 and .81 except for Cluster Centroids which had a poor score of .58. This tells me that these models are average at producing fewer false positives or false negatives.

The objective is to flag high-risk loans. The recall values are poor for both models as they are between .41-.68 for both the high-risk and low-risk detection. This results in a high count of flase negatives. All models obtained a low precision output of .1 for high-risk loans and great, low F1 scores (.1-.2) for high-risk loans, which can allow for more inaccurate predictions. This is also shown by low balanced accuracy scores. 

## Extension

### Balanced Accuracy Scores
Balanced Random Forest Classifier: 0.78

Easy Ensemble AdaBoost: 0.93

### Confusion Matrices
Balanced Random Forest Classifier

True Positive= 58 | False Positive= 29

False Negative= 1,560 | True Negative= 15,558

Easy Ensemble AdaBoost

True Positive= 79 | False Positive= 8

False Negative= 979 | True Negative= 16,139

### Describing Precision and Recall Scores
Balanced Random Forest Classifier (BRFC)

Precision (high-risk): 0.04 Precision (low-risk): 1.00

Recall (high-risk): 0.67 Recall (low-risk): 0.91

### Easy Ensemble AdaBoost

Precision (high-risk): 0.07 Precision (low-risk): 1.00

Recall (high-risk): 0.91 Recall (low-risk): 0.94

### Summary
Both of these models fit a perfect precision score for low-risk detections and great F1 scores between .95 and .97 meaning that the models are good at producing few false negatives and false positives. 

The objective is to flag high-risk loans and the recall values for both models, high-risk and low-risk detections, were high as well, .91-.94 (except for the Balanced Random Forest Classifier which had a recall value for high-risk detections of .67- which is poor). This is a significant drop in flase negatievs compared to the models reviewed earlier. These two models also have higher precision outputs for high-risk loans of .04 and .07 with F1 scores of .07 and .14 for high-risk loans which will allow for accurate predictions. 

## Recommendations
The strongest predictor model is the Easy Ensemble AdaBoost Classifier since it has the highest accuracy score, .91, highest F1 score for high-risk and low risk loans. Even though this model is the strongest, it still shows some potential for false positives because of its high recall value of .91 and its low precision value of .07. This means that some low-risk loans could be rejected unfairly because they may be labeled as high-risk. This type of miscalculation in a model has the potential to be inefficient and costly for a firm in the long run if they have to go back and double check their work. 
