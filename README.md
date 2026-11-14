# Loan Rejection Classifier – Improving a Model on a Rare Class

## Dataset
- **Name:** Loan Prediction Problem Dataset (training file, 614 rows × 13 columns)
- **Source:** Kaggle – https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
- **File in this repo:** `loan_data.csv`
- **Target:** `Loan_Status` → rare class = **rejected (N)**, 192 of 614 rows (≈ 31%)

## Summary
1. A default Random Forest reaches 77% accuracy, but in cross-validation it misses **53%** of the rejected loans, because it relies almost only on `Credit_History`.
2. Error analysis showed that missed rejections have good credit history but loans that are large compared with income. So I added ratio features (total income, EMI, loan-to-income), then `class_weight="balanced"`, then a lower threshold (0.4). I chose all of these with 5-fold CV on the training set only.
3. Main metric: **recall of the rejected class**, with the rule precision ≥ 0.50. A missed bad loan costs the bank much more than a false alarm.

## Before / after (test set, rejected class)

| Version | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Dummy (most frequent) | 0.691 | 0.000 | 0.000 | 0.000 |
| V1 Baseline Random Forest | 0.780 | 0.690 | 0.526 | 0.597 |
| V2 + new ratio features | 0.772 | 0.656 | 0.553 | 0.600 |
| V3 + class_weight="balanced" | 0.748 | 0.600 | 0.553 | 0.575 |
| **V4 Final (threshold 0.4)** | 0.691 | 0.500 | **0.658** | 0.568 |

The final model catches **25 of 38** rejections instead of 20, so 5 fewer bad loans are missed. The trade-off is 16 more false alarms (9 → 25). In real use, flagged applications would go to a loan officer for manual review.

## How to run
```bash
git clone https://github.com/Moiz-Mukhtar/loan-rejection-classifier.git
cd loan-rejection-classifier
pip install -r requirements.txt
jupyter notebook loan_model_improvement.ipynb
```
Then choose **Kernel → Restart & Run All**. The notebook saves its charts in the `figures/` folder.

## Files
- `loan_model_improvement.ipynb` – the full notebook (Steps 1–7)
- `loan_data.csv` – dataset
- `requirements.txt` – Python packages
- `figures/` – charts used in the report
