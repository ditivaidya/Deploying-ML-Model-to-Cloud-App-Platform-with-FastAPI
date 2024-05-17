# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is an XGBoost Binary Classfier model with the following tuned hyper parameters -

||Hyperparameters|Value|
|---|---|---|
 1|*colsample_bytree*|1.0|
 2|*gamma*|0.1|
 3|*learning_rate*|0.1|
 4|*max_depth*|7|
 5|*min_child_weight*|1|
 6|*scale_pos_weight*|3.0135|
 7|*subsample*|0.8|

## Intended Use

This model used to predict whether a person makes more than $50K a year based on a few features (See link below to read about features) 

## Training Data

The model is trained on the census data from the UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/20/census+income

## Evaluation Data

The model was evaluated using an 80-20, train/test split.

## Metrics

- Precision Score 0.611
- Recall Score 0.832
- fbeta 0.704

## Ethical Considerations

This model does not check feature imoprtance. This could lead to ethical concerns such as biases based on features such as gender or race. Based on what this model is used for, this can be a serious ethical concern.

## Caveats and Recommendations

It is recomended that the model is monitored frequenty and new data is analysed to assess changing demographics and minimise Data Drift.
It is also recomended that data scientists look into feature importance to avoid any unconscious biases.