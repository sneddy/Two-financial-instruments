# Two financial instruments prediction


### Data
The attached zip file contains a CSV file, containing price data for two financial instruments, called X and Y.
These are real market prices, sampled once every 10 seconds.

The four columns are 

- "timestamp" - the timestamp of this observation (millis since 00:00:00 January 1, 1970)

- "xprice" - the price of instrument X

- "yprice" - the price of instrument Y

- "returns" - the returns of instrument Y

The 'returns' column is defined to be the price ten minutes from now, minus the current price. 
It has been truncated to remove outliers. In the event that there is no price ten minutes from now 
(because the market has closed, for example) then the latest price available is used.

## Task

The task is to build a model to predict the returns of asset Y using lagged price information of X and Y.
The model will be judged on out-of-sample mean-squared prediction error, defined as:

```latex
MSPE = mean((returns - prediction) ^ 2)
```

To make the forecast at time t, you may use data up to and including time t.

## Decision requirements 

The solution should be in the form of two functions

parameters = modelEstimate(trainingFilename)
predictions = modelForecast(testFilename, parameters)
 
The "modelEstimate" function takes the filename of a CSV file containing the training data, 
and returns an object containing the parameters of the fitted model.

The "modelForecast" function takes the filename of a CSV file containing the test data and a set of parameters,
and outputs a vector of predictions (one prediction for each observation in the data set).

## Launch
Suppose the files train.csv and test.csv are in the root folder. Then you can run bash command
```nash
sh run.sh
```
or 
```bash
python modelEstimate.py train.csv
python modelForecast.py train.csv test.csv
```
With default parameters:
- modelEstimate.py will train ridge model and store to ridge_weights.model
- modelForecast.py will predict ridge model and store result to predicted.npy

I pass the parameter test.csv to modelForecast.py only for calculating features, not for cheating :)

## File structure
    ├── FeatureSelection.ipynb  - experiments with feature selection
    ├── FinalHeap.ipynb         - experiments with building features
    ├── feature_extractor.py    - script for building selected pack of features
    ├── features_config.py      - global parameters of training and feature extraction
    ├── helper.py      -        - simple helper function for printing weights of features
    ├── modelEstimate.py        - main function for fitting of linear model
    ├── modelForecast.py        - main function for forecasting of linear model
    ├── ts_features.py          - basic features constructors
    └── ts_validation.py        - different methods of validation
 
 
 ## Features 
From xprice and yprice I build 6 main features - and the other features were obtained as various kinds of aggregations and moving statistics above them. 

The total number of features was about 700

**Main features:**
- log_features (log x, log y)
- spread_features (x - y)
- relation_features (x / y)
- geom_features ((x * y) ^ 0.5)
- square_features ((x ^ 2 + y ^ 2) / 2) ^ 0.5
- garmonic_features (2 / (1 / x + 1 / y))


 
 ## Validation
 I split half of my data to 5 segments:
 - Train from 0 to 50% - validation from 50% to 60%
 - Train from 0 to 60% - validation from 60% to 70%
 - Train from 0 to 70% - validation from 70% to 80%
 - Train from 0 to 80% - validation from 80% to 90%
 - Train from 0 to 90% - validation from 90% to 100%
 
 After that I calculate 5 number: r2 score, multiplied by 100 for each segment. 
 
 **My validation strategy:** I tried to maximize the average **and** minimum among these numbers
 
 My final metrics on validation:
 - validation from 50% to 60%:  1.73
 - validation from 50% to 70%:  3.69 
 - validation from 50% to 80%:  0.83
 - validation from 50% to 90%:  4.83
 - validation from 50% to 100%: 5.42
 - min  score on validation:    0.83  
 - mean score on validation:    3.25
 
 I used this scheme of validation and add-del strategy for greedy features selection.
 
 The final number of features has been reduced to 42.
 
 
 
 
 
