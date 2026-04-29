# income_prediction
Binary income classification using supervised machine learning on US Census data.

Goal: Predict whether an individual's annual income is above $50k based on demographic and employment features.

## Dataset

2 datasets were used and compared:

- **'1994 Adult Census Dataset'** - Widely used ML dataset derived from 1994 U.S. Census Bureau database.
                              - 48,842 records, 14 features
- **'2020 ASEC (Annual Social and Economic Supplement)'** - Raw Census data cleaned and filtered to match 1994 dataset.
                                                      - 157,959 records reduced to 38,576
- **Features**: "Age", "Education", "Occupation", "Hours per week", "Capital gain", "Capital loss", "Workclass", "Marital Status", "Relationship", "Race", "Gender", "Native Country"

## Approach

Three different models were implimented:

1. **Linear Regression** - Elstablishes performance floor using reguralized linear methods (Ridge, Lasso, ElasticNet, Logistic Regression)
2. **Gradient Boosted Tree** - 'GradientBoostingClassifier' with hyperparameter tuning via 'RandomizedSearchCV'
3. **Stacking Ensemble** - Six base models (Logistic Regression, Random Forest, GBT, MLP, Catboost, XGBoost) combined with Logistic Regression meta-learner

## Results

| Model | Dataset | ROC-AUC |
  |---|---|---|
  | Linear Regression | 1994 | 0.821 |
  | Gradient Boosted Tree | 2020 | 0.862 |
  | Gradient Boosted Tree | 1994 | **0.927** |
  | Stacking Ensemble | 1994 | 0.720 |

  **Key Findings:** The Gradient Boosted Tree outperformed the stacked model despite using fewer models. The stacking model's underperformance is attributed to convergence issues in the MLP and Logistic Regression models, showing that more models does not always lead to better performance.

## Tech Stack

  - Python, Jupyter Notebook
  - scikit-learn, XGBoost, CatBppst
  - pandas, NumPy, matplotlib

## How To Run
  1. Clone Repo
  2. Install Dependencies: 'pip install scikitlearn xgboost catboost pandas numpy jupyter'
  3. Open Notebooks in order: data cleaning - baseline - GBT - stacking
