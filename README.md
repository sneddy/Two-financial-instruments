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

![equation](MSPE = mean((returns - prediction) ^ 2))

To make the forecast at time t, you may use data up to and including time t.

## Solution 

The solution should be in the form of two functions

parameters = modelEstimate(trainingFilename)
predictions = modelForecast(testFilename, parameters)
 
The "modelEstimate" function takes the filename of a CSV file containing the training data, 
and returns an object containing the parameters of the fitted model.

The "modelForecast" function takes the filename of a CSV file containing the test data and a set of parameters,
and outputs a vector of predictions (one prediction for each observation in the data set).

 
