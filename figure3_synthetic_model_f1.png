# Modular Workflow for Cybersecurity Education

This repository implements a modular experimental workflow for payload-free intrusion detection. The goal is to support reproducible learning in data and communication security, not to provide a closed operational IDS product.

## Workflow stages

1. **Input dataset**
   - Controlled synthetic IDS data
   - CICIDS-style public flow-level CSV files
   - User-supplied flow-level CSV files

2. **Preprocessing**
   - Label normalization
   - Removal of identifiers and timestamps
   - Missing-value and infinite-value handling
   - Feature scaling

3. **Feature handling**
   - Payload-free numerical flow features
   - Feature grouping for ablation when names support semantic grouping

4. **Classifier module**
   - Logistic Regression
   - KNN
   - Naive Bayes
   - Linear SVM
   - MLP
   - Random Forest
   - XGBoost
   - Optional PCA and K-Means anomaly baselines

5. **Evaluation module**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - PR-AUC
   - Training and scoring time

6. **Visualization and interpretation**
   - ROC and precision-recall curves
   - F1-score comparison
   - Feature-group ablation
   - Robustness testing under perturbation
   - Optional SHAP summary plots

## Educational use

The modular design lets learners ask practical questions:

- How do linear, distance-based, probabilistic, neural, bagging, and boosting models differ on the same IDS task?
- Why can ROC-AUC and PR-AUC tell different stories under class imbalance?
- Which feature groups carry more useful signal?
- How stable is a model when feature measurements are perturbed?
- How can explainability tools such as SHAP help interpret model behavior?

The current experiments serve as case studies demonstrating this workflow.
