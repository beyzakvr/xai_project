## Overview
This project applies Explainable AI (XAI) methods to a machine learning model that predicts user skepticism towards AI systems. The goal of the project is to compare explanation methods and evaluate explanation quality using metrics such as faithfulness, stability, and completeness.

## Dataset
The dataset used in this project is the **AI Skepticism Dataset**, which contains behavioral, trust-related, and decision-making features. The target variable is:

`user_skepticism_category`

Classes:
- Blind Trust
- Moderate Trust
- Skeptical
- Highly Skeptical

Features include:
- Trust score
- Verification duration
- Fact-checking behavior
- Decision importance
- Answer accuracy
- Confidence percentage
- Belief alignment
- Fact-check method

## Model
A **Random Forest classifier** was used because it performs well on tabular data and mixed feature types.

Preprocessing steps:
- Missing value imputation
- Standardization of numeric features
- One-hot encoding of categorical features
- Train-test split (80/20)

## Explanation Methods
Two post-hoc explanation methods were used:

### SHAP
- Global feature importance
- Local feature explanations
- SHAP summary plot

### LIME
- Local explanation for individual predictions
- Rule-based explanation output

## Explanation Evaluation
The explanation methods were evaluated using the following metrics:

### Faithfulness
Important features were removed and the decrease in model confidence was measured.

### Stability
Small noise was added to input data and explanation rankings were compared using Spearman rank correlation.

### Completeness
Completeness was discussed theoretically using SHAP’s additive feature attribution property.

## Results
Model performance:
- Accuracy: 0.97
- Macro F1 Score: 0.94

Explanation evaluation:
- SHAP Faithfulness: 0.38
- SHAP Stability: 0.99

The results indicate that SHAP explanations were both faithful and highly stable.

