# Confusion Matrix Report for Healthcare Classification Model

## 1. Details
This report explains how a machine learning model was evaluated using a confusion matrix in a healthcare system. The model was developed to classify patients into two categories: **Healthy** and **Sick**, based on medical test results and symptoms.

---

## 2. Dataset Description
The dataset contains **100 patient records** collected from routine health screenings.

- 70 records were used for training the model  
- 30 records were used for testing the model  

The testing dataset is important because it uses unseen data to check how well the model works in real situations.

---

## 3. Model Testing Result
After training, the model was tested using the 30 patient records.

During testing:
- The model made **3 incorrect predictions**
- These errors include:
  - False Negative (FN): Sick patients predicted as Healthy  
  - False Positive (FP): Healthy patients predicted as Sick  

Assumption:
- Sick patients = 15  
- Healthy patients = 15  

Results:
- False Negative (FN) = 2  
- False Positive (FP) = 1  
- True Positive (TP) = 13  
- True Negative (TN) = 14  

---

## 4. Confusion Matrix Table

|                      | Predicted Sick | Predicted Healthy |
|----------------------|---------------|-------------------|
| **Actual Sick**      | 13 (TP)       | 2 (FN)            |
| **Actual Healthy**   | 1 (FP)        | 14 (TN)           |

---

## 5. Explanation

- **True Positive (TP = 13)**  
  Sick patients correctly identified as sick  

- **True Negative (TN = 14)**  
  Healthy patients correctly identified as healthy  

- **False Positive (FP = 1)**  
  Healthy patient wrongly predicted as sick  
  → May cause unnecessary stress or treatment  

- **False Negative (FN = 2)**  
  Sick patients wrongly predicted as healthy  
  → Very serious because illness is missed  

---

## 6. Model Performance

- **Accuracy**  
  = (TP + TN) / Total  
  = (13 + 14) / 30  
  = **90%**

- **Precision**  
  = TP / (TP + FP)  
  = 13 / 14  
  = **92.86%**

- **Recall (Sensitivity)**  
  = TP / (TP + FN)  
  = 13 / 15  
  = **86.67%**

---

## 7. Important Analysis

The model has a good accuracy of **90%**, meaning most predictions are correct.

However, in healthcare systems:
- False Negative is very dangerous  
- Missing a sick patient can lead to serious health risks  

Even with high accuracy, the model still needs improvement to reduce these risks.

---

## 8. Conclusion

The confusion matrix helps clearly show how the model performs.

- The model works well overall  
- But it still misses some sick patients  

Future improvement should focus on:
- Increasing recall  
- Reducing false negatives  

This is very important for real-world healthcare applications.
