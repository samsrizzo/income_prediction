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
3. **Stacking Ensemble** - 
