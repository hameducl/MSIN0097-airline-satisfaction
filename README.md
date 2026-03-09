# Airline Passenger Satisfaction Prediction
### MSIN0097 Predictive Analytics — Individual Coursework 2025–26

---

## Project Overview

**Research Question:** Can passenger demographics, travel characteristics, and in-flight service ratings predict whether a passenger will be satisfied with their airline experience?

This project builds an end-to-end binary classification pipeline using 129,880 real airline passenger survey responses. The target variable is `satisfaction` (satisfied vs neutral or dissatisfied).

---

## Repository Structure

```
airline_project/
├── airline_satisfaction.ipynb   # Main notebook (all 6 steps)
├── train.csv                    # Training data (103,904 rows) — download from Kaggle
├── test.csv                     # Test data (25,976 rows) — download from Kaggle
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── artefacts/                   # Generated after running notebook
    ├── final_xgboost_model.pkl
    ├── feature_scaler.pkl
    ├── features.txt
    └── test_predictions.csv
```

---

## Data Source

**Kaggle:** [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) by TJ Klein

Download both `train.csv` and `test.csv` and place them in the same folder as the notebook.

---

## How to Run

### Option 1: VS Code with Anaconda (Recommended)
1. Place `train.csv` and `test.csv` in the project folder
2. Open `airline_satisfaction.ipynb` in VS Code
3. Select kernel: `base (Python 3.x) Conda Env`
4. Open terminal and run: `pip install xgboost shap`
5. Click **Run All**

### Option 2: Google Colab
1. Upload notebook + both CSV files to Colab
2. Run all cells (first cell installs xgboost and shap automatically)

---

## Methodology Summary

| Step | Description |
|------|-------------|
| 1 | Problem framing: binary classification, service ratings as features |
| 2 | EDA: class balance, satisfaction by class/travel type, service distributions |
| 3 | Data prep: median imputation, binary encoding, one-hot encoding, feature engineering |
| 4 | Model exploration: Logistic Regression → Random Forest → XGBoost → Neural Net |
| 5 | Fine-tuning: CV ablation, test evaluation, SHAP interpretability, calibration |
| 6 | Final solution: model card, limitations, artefact saving |

---

## Key Design Decisions

- **Stratified random split** (not temporal) — appropriate since passenger records have no time ordering
- **Median imputation** for missing Arrival Delay — robust to right-skewed delay distributions
- **ROC-AUC** as primary metric — threshold-independent, handles mild class imbalance
- **SHAP** for interpretability — identifies which service dimensions drive satisfaction most

---

## Agent Tooling Attribution

This project used **Claude (Anthropic)** as an agent collaborator for notebook scaffolding, EDA templates, feature engineering suggestions, and model pipeline code. All outputs were reviewed and verified. See the Agent Usage Log + Decision Register in the report appendix.

---

## References

- Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- Lundberg, S.M. and Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
- TJ Klein (2020). Airline Passenger Satisfaction. Kaggle. Available at: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
