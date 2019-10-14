import numpy as np


def average_true_range(data, trend_periods=7, ma_type='ewma'):
    for index, row in data.iterrows():
        prices = [row['high'], row['low'], row['close'], row['open']]
        if index > 0:
            val1 = np.amax(prices) - np.amin(prices)
            val2 = abs(np.amax(prices) - data.at[index - 1, 'close'])
            val3 = abs(np.amin(prices) - data.at[index - 1, 'close'])
            true_range = np.amax([val1, val2, val3])
        else:
            true_range = np.amax(prices) - np.amin(prices)

        data.at[index, 'true_range'] = true_range
    
    weights = np.array(list(reversed([(i+1) for i in range(trend_periods)])))
    sum_weights = np.sum(weights)

    if ma_type == 'wma':
        #data['atr'] = (
        #    data['true_range']
        #    .rolling(window=trend_periods)
        #    .apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)
        #)
        data['atr'] = data['true_range'].rolling(len(weights)).apply(lambda x: np.correlate(x,weights/sum(weights)))
    elif ma_type == 'ewma':
        data['atr'] = data['true_range'].ewm(alpha=(1/trend_periods), min_periods=trend_periods, adjust=True, ignore_na=True, axis=0).mean()
    else:
        raise Exception()

    return data


def super_trend(data, factor=3, trend_periods=7, up_fields=['high', 'low'], down_fields=['high', 'low'], smooth=False):
    if smooth:
        weights = np.array(list(reversed([(i+1) for i in range(trend_periods)])))
        sum_weights = np.sum(weights)
        
        #data['weighted_ATR'] = (
        #    data['atr']
        #    .rolling(window=trend_periods)
        #    .apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)
        #)
        data['weighted_ATR'] = data['atr'].rolling(len(weights)).apply(lambda x: np.correlate(x,weights/sum(weights)))
                       
        data['Up'] = data[up_fields].mean(axis=1) - factor * data['weighted_ATR']
        data['Dn'] = data[down_fields].mean(axis=1) + factor * data['weighted_ATR']

    else:
        data['Up'] = data[up_fields].mean(axis=1) - factor * data['atr']
        data['Dn'] = data[down_fields].mean(axis=1) + factor * data['atr']

    return data


def double_super(df, tup_fields=['close'], tdown_fields=['close'], trend_fields=['close']):
    df['Tup'] = np.nan
    df['Tup_check'] = df[tup_fields].mean(axis=1)

    for index, row in df.iterrows():
        if index > 0:
            if df.at[index - 1, 'Tup_check'] > df.at[index - 1, 'Tup']:
                df.at[index, 'Tup'] = max(df.at[index, 'Up'], df.at[index - 1, 'Tup'])
            else:
                df.at[index, 'Tup'] = df.at[index, 'Up']

    df['Tdown'] = np.nan
    df['Tdown_check'] = df[tdown_fields].mean(axis=1)

    for index, row in df.iterrows():
        if index > 0:
            if df.at[index - 1, 'Tdown_check'] < df.at[index - 1, 'Tdown']:
                df.at[index, 'Tdown'] = min(df.at[index, 'Dn'], df.at[index - 1, 'Tdown'])
            else:
                df.at[index, 'Tdown'] = df.at[index, 'Dn']
        else:
            continue

    df['Trend'] = np.nan
    df['Trend_check'] = df[trend_fields].mean(axis=1)

    for index, row in df.iterrows():
        if index > 0:
            if df.at[index, 'Trend_check'] > df.at[index - 1, 'Tdown']:
                df.at[index, 'Trend'] = 1

            elif df.at[index, 'Trend_check'] < df.at[index - 1, 'Tup']:
                df.at[index, 'Trend'] = -1

            else:
                df.at[index, 'Trend'] = df.at[index - 1, 'Trend']

    df['Tsl1'] = 0
    df['Tsl1'] = df['Tsl1'].astype(float)

    for index, row in df.iterrows():
        if index > 0:
            if df.at[index, 'Trend'] == 1:
                df.at[index, 'Tsl1'] = df.at[index, 'Tup']
            else:
                df.at[index, 'Tsl1'] = df.at[index, 'Tdown']
        else:
            continue

    df['Tsl2'] = 0
    df['Tsl2'] = df['Tsl2'].astype(float)

    for index, row in df.iterrows():
        if index > 0:
            if df.at[index, 'Trend'] == 1:
                df.at[index, 'Tsl2'] = df.at[index, 'Tdown']
            else:
                df.at[index, 'Tsl2'] = df.at[index, 'Tup']
        else:
            continue

    return df


# def atr_goes_green(df):
#     return df.iloc[-2]['Trend'] == -1 and df.iloc[-1]['Trend'] == 1


# def atr_is_green(df):
#     return df.iloc[-1]['Trend'] == 1


# def atr_goes_red(df):
#     return df.iloc[-2]['Trend'] == 1 and df.iloc[-1]['Trend'] == -1


# def atr_is_red(df):
#     return df.iloc[-1]['Trend'] == -1
