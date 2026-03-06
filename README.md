# Credit Card Fraud Detection

ML project for my data science course. Trying to detect fraud in credit card transactions.

The dataset is super imbalanced - only 0.17% of transactions are actually fraud. So a model that just says "not fraud" every time already gets 99.83% accuracy, which is obviously useless. Needed to use proper metrics like AUPRC instead of accuracy.

## Dataset

[Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - ~284k transactions from European cardholders over 2 days in 2013. Only 492 are fraud.

- `V1`-`V28` are PCA features (anonymized)
- `Time` and `Amount` are the only original features
- `Class` is the target (0 = legit, 1 = fraud)

## What's in this

Cleaned and explored the data, scaled the non-PCA features, then used SMOTE to deal with the class imbalance (applied after splitting so no data leakage). Trained Logistic Regression, Random Forest, and XGBoost, compared them with PR curves and confusion matrices, tuned the threshold, and ran cross-validation to make sure results hold up.

## How to run

```
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), drop it in this folder, then run the notebook.

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, XGBoost

## Results

XGBoost and Random Forest did the best (by AUPRC). Threshold tuning helped squeeze out a bit more F1. Check the notebook for the full comparison.
