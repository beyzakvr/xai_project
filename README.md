# Explainable AI Project – AI Skepticism Classification

## Overview
This project applies Explainable AI (XAI) methods to a machine learning model that predicts user skepticism towards AI systems. The goal is to compare explanation methods and rigorously evaluate explanation quality using faithfulness, stability, and completeness metrics.

## Dataset
The dataset used in this project is the **AI Skepticism Dataset**, which contains behavioral, trust-related, and decision-making features.

**Target variable:**
- `user_skepticism_category`

**Classes:**
- Blind Trust
- Moderate Trust
- Skeptical
- Highly Skeptical

**Example features:**
- Trust score
- Verification duration
- Fact-checking behavior
- Decision importance
- Answer accuracy
- Confidence percentage
- Belief alignment
- Fact-check method

## Model
A **Random Forest classifier** was used as the primary model due to its strong performance on tabular data.

A Logistic Regression model was also trained as a baseline.

**Preprocessing steps:**
- Missing value imputation
- Standardization of numeric features
- One-hot encoding of categorical features
- Train-test split (80/20)

## Explanation Methods
Three XAI approaches were used:

### SHAP
- Global feature importance
- Local explanations per instance
- SHAP summary plots
- Quantitative evaluation (faithfulness, stability, F-Fidelity)

### LIME
- Local rule-based explanations
- Feature importance for individual predictions
- Quantitative comparison with SHAP (faithfulness + stability)

### MCD (Multi-dimensional Concept Discovery – inspired)
- Concept-based analysis using Random Forest internal representation
- Leaf-index embeddings used as a proxy feature space
- KMeans clustering to discover concept groups
- PCA used to construct multi-dimensional concept subspaces
- Completeness evaluated via concept coverage of representation

## Explanation Evaluation
The explanation methods were evaluated using multiple complementary metrics:

### Faithfulness
- **Deletion test:** Remove top-k important features and measure drop in prediction confidence
- **Insertion test:** Keep only important features and measure how much prediction is recovered

### F-Fidelity (inspired)
- Distribution-aware masking instead of zeroing features
- Important features replaced with values sampled from training data
- Provides a more realistic estimate of faithfulness

### Stability
- Gaussian noise added to inputs
- Spearman rank correlation used to compare explanation consistency

### Completeness
- **SHAP additivity:** verifies that feature contributions sum to prediction
- **MCD-inspired concept completeness:**
  - Representation projected onto concept subspaces
  - Completeness measured as fraction of representation explained by concepts
  - Concept relevance used to analyze contribution of each subspace

## Results

### Model Performance
- Accuracy: **0.97**
- Macro F1 Score: **0.94**

### Explanation Evaluation
| Method | Deletion | Insertion | F-Fidelity | Stability |
|--------|--------|----------|------------|-----------|
| SHAP   | 0.3823 | 0.3513   | 0.4361     | 0.9948    |
| LIME   | 0.2905 | 0.1757   | 0.3222     | 0.3970    |

### Key Findings
- SHAP consistently outperformed LIME in faithfulness and stability
- LIME provides intuitive explanations but lacks consistency
- F-Fidelity provides a more robust faithfulness estimate
- MCD-inspired analysis shows that model behavior can be interpreted at a concept level

## Visual Outputs
The project generates the following outputs:

- `shap_summary.png` → global SHAP importance
- `lime_explanation.html` → local LIME explanation
- `xai_comparison.png` → SHAP vs LIME comparison
- `mcd_concept_analysis.png` → concept relevance and completeness distribution

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the project:
```bash
python main.py
```

## Repository Structure
```
Final Project/
│
├── main.py
├── ai_skepticism_dataset.csv
├── requirements.txt
├── README.md
├── shap_summary.png
├── lime_explanation.html
├── xai_comparison.png
├── mcd_concept_analysis.png
```

## Notes
- SHAP is the most reliable explanation method for this task
- LIME is useful for interpretability but less stable
- F-Fidelity improves faithfulness evaluation by avoiding unrealistic perturbations
- MCD is implemented as a simplified, concept-based extension

## Author
Beyzanur Kıvır
Explainable AI Course Project
