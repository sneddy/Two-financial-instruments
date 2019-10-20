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
I pass the parameter test.csv to modelForecast.py only for calculating features, not for cheating :)

## File structure
  ├── FeatureSelection.ipynb - experiments with feature selection
  ├── FinalHeap.ipynb - experiments with building features
  ├── some_experiment
  ├── train_config.yaml
  ├── inference_config.yaml
  └── requirements.txt
 
