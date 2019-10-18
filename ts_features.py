import pandas as pd
import numpy as np
from datetime import datetime


def init_data(fname):
    data = pd.read_csv('data.csv')

    data['yx_spread'] = data.yprice - data.xprice
    data['yx_relation'] = data.yprice  / data.xprice
    data['xy_relation'] = data.xprice / data.yprice
    data['xy_geom'] = np.sqrt(data.xprice * data.yprice)
    data['xy_garmonic'] = 2 / (1 / data.xprice + 1 / data.yprice)
    
#     data.xprice = (data.xprice - data.xprice.min())# / data.xprice.std() 
#     data.yprice = (data.yprice - data.yprice.min())# / data.yprice.std() 
    data['timestamp'] = data['timestamp'] // 1000
    data['timestamp'] = data['timestamp'].apply(lambda stamp: datetime.fromtimestamp(stamp))
    data['timestamp'] = data['timestamp'] - pd.Timedelta(hours=1) # for flexibility
    data.index = data['timestamp']
    
    data['weekday'] = data.timestamp.dt.weekday
    data['day'] = (data.timestamp.dt.date - data.timestamp.dt.date.min()).apply(lambda x: int(x.days))
    day_close_time = data.day.map(data.groupby('day').timestamp.max())
    data['periods_before_closing'] = (day_close_time - data.timestamp).apply(lambda x: x.seconds // 10)
    day_open_time = data.day.map(data.groupby('day').timestamp.min())
    data['periods_after_opening'] = (data.timestamp - day_open_time).apply(lambda x: x.seconds // 10)
#     data.drop('timestamp', 1, inplace=True)
    return data

def init_norm_data(fname, min_train_ratio=0.1, eps=1e-5):
    data = pd.read_csv('data.csv')
    
    min_train_rows = int(data.shape[0] * min_train_ratio)
    data.yprice = (data.yprice - data.yprice[:min_train_rows].min()) / \
        (data.yprice[:min_train_rows].max() - data.yprice[:min_train_rows].min())
    data.xprice = (data.xprice - data.xprice[:min_train_rows].min()) / \
        (data.xprice[:min_train_rows].max() - data.xprice[:min_train_rows].min())
    
    data['yx_spread'] = data.yprice - data.xprice
    ynorm = data.yprice - data.yprice.head(1000).mean()
    data['yx_relation'] = data.yprice  / (data.xprice + eps)
    data['xy_relation'] = data.xprice / (data.yprice + eps)
    data['xy_geom'] = np.sqrt(data.xprice * data.yprice)
    data['xy_garmonic'] = 2 / (1 / data.xprice + 1 / data.yprice)
    
#     data.xprice = (data.xprice - data.xprice.min())# / data.xprice.std() 
#     data.yprice = (data.yprice - data.yprice.min())# / data.yprice.std() 
    data['timestamp'] = data['timestamp'] // 1000
    data['timestamp'] = data['timestamp'].apply(lambda stamp: datetime.fromtimestamp(stamp))
    data['timestamp'] = data['timestamp'] - pd.Timedelta(hours=1) # for flexibility
    data.index = data['timestamp']
    
    data['weekday'] = data.timestamp.dt.weekday
    data['day'] = (data.timestamp.dt.date - data.timestamp.dt.date.min()).apply(lambda x: int(x.days))
    day_close_time = data.day.map(data.groupby('day').timestamp.max())
    data['periods_before_closing'] = (day_close_time - data.timestamp).apply(lambda x: x.seconds // 10)
    day_open_time = data.day.map(data.groupby('day').timestamp.min())
    data['periods_after_opening'] = (data.timestamp - day_open_time).apply(lambda x: x.seconds // 10)
#     data.drop('timestamp', 1, inplace=True)
    return data


def add_diffs(df, column, uselags):
    new_columns = []
    for lag in uselags:
        colname = '{}_diff_{}'.format(column, lag)
        df.loc[:, colname] = df[column].diff(lag)
        new_columns.append(colname)
    print(new_columns)
    return new_columns

def add_shifts(df, column, uselags):
    new_columns = []
    for lag in uselags:
        colname = '{}_lag_{}'.format(column, lag)
        df.loc[:, colname] = df[column].shift(lag).values
        new_columns.append(colname)
    print(new_columns)
    return new_columns

def add_rolling_mean(df, column, windows):
    new_columns = []
    for window_size in windows:
        colname = '{}_ma_{}'.format(column, window_size)
        df.loc[:, colname] = df[column].rolling(window=window_size).mean()
        new_columns.append(colname)
    print(new_columns)
    return new_columns

def add_curstom_rolling_operation(df, column, agg_function, function_name, windows):
    new_columns = []
    for window_size in windows:
        colname = '{}_{}_{}'.format(column, function_name, window_size)
        df.loc[:, colname] = df[column].rolling(window=window_size).agg(agg_function)
        new_columns.append(colname)
    print(new_columns)
    return new_columns  

def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def add_rsi(df, column, windows):
    new_columns = []
    for window_size in windows:
        colname = '{}_rsi_{}'.format(column, window_size)
        df.loc[:, colname] = rsiFunc(df[column].values, window_size)
        new_columns.append(colname)
    print(new_columns)
    return new_columns  

def add_ewma(df, column, halfsize_windows):
    new_columns = []
    for window_size in halfsize_windows:
        colname = '{}_ewma_{}'.format(column, window_size)
        ewm = pd.Series.ewm(df[column].drop_index(), halflife=window_size).mean().values
        df.loc[:, colname] = ewm
        new_columns.append(colname)
    print(new_columns)
    return new_columns

def add_intraday_ewma(df, column, halfsize_windows):
    new_columns = []
    days = df.day.unique()
    for day in days:
        df_mask = (df.day == day)
        for window_size in halfsize_windows:
            colname = '{}_dayly_ewma_{}'.format(column, window_size)
            ewm = pd.Series.ewm(df.loc[df_mask, column], halflife=window_size).mean().values
            df.loc[df_mask, colname] = ewm
    new_columns = ['{}_dayly_ewma_{}'.format(column, window_size) for window_size in halfsize_windows]
    print(new_columns)
    return new_columns  

def add_time_depended_rolling(df, source_column, windows, agg_fun, agg_repr):
    '''
        df: source dataframe
        source_column: column for building feature
        windows: list with periods (1 period = 10 sec)
        agg_fun: aggregation function
        agg_repr: name of agg function
    '''    
    new_cols = []
    for agg_period in windows:
        agg_shifts = range(10, agg_period * 10, 10)
        period_repr = '{}s'.format(agg_period * 10)
        
        agg_helper_df = df[source_column].resample(
            period_repr, label='right', closed='right').agg(agg_fun)
                                             
        for shift in agg_shifts:
            agg_helper_df = agg_helper_df.append(df[source_column].resample(
                period_repr, label='right', closed='right', base=shift).agg(agg_fun))
        colname = '{}_time_{}_{}'.format(source_column, agg_repr, agg_period)
        df.loc[:, colname] = agg_helper_df
        new_cols.append(colname)
    print(new_cols)
    return new_cols

def add_time_depended_dif(df, column, windows):
    pass

def add_hand_feats(df, eps=1e-5):
    close_price_per_day = df.groupby('day').timestamp.max().shift(1).map(
        df[['timestamp', 'yprice']].set_index('timestamp').yprice)
    y_mapped = df.day.map(close_price_per_day)
    
    df.loc[:, 'ydiff_from_closing'] = (df.yprice - y_mapped).fillna(0)
    df.loc[:, 'yrel_from_closing'] = (df.yprice / (y_mapped + eps)).fillna(1)
    
    close_price_per_day = df.groupby('day').timestamp.max().shift(1).map(
        df[['timestamp', 'xprice']].set_index('timestamp').xprice)
    x_mapped = df.day.map(close_price_per_day)
    df.loc[:, 'xdiff_from_closing'] = (df.xprice - x_mapped).fillna(0)
    df.loc[:, 'xrel_from_closing'] = (df.xprice / (x_mapped + eps)).fillna(1)
    
    open_price_per_day = df.groupby('day').timestamp.min().map(
        df[['timestamp', 'yprice']].set_index('timestamp').yprice)
    y_mapped = df.day.map(open_price_per_day)
    df.loc[:, 'ydiff_from_opening'] = df.yprice - y_mapped
    df.loc[:, 'yrel_from_opening'] = df.yprice / (y_mapped + eps)
    
    open_price_per_day = df.groupby('day').timestamp.min().map(
        df[['timestamp', 'xprice']].set_index('timestamp').xprice)
    x_mapped = df.day.map(open_price_per_day)
    df.loc[:, 'xdiff_from_opening'] = df.xprice - x_mapped
    df.loc[:, 'xrel_from_opening'] = df.xprice / (x_mapped + eps)
    
    new_columns = [
        'ydiff_from_closing', 'xdiff_from_closing',
        'yrel_from_closing', 'xrel_from_closing',
        'ydiff_from_closing', 'xdiff_from_closing',
        'yrel_from_opening', 'xrel_from_opening'
    ]
    print(new_columns)
    return new_columns

def add_full_history_diff(df, col):
    mean = df[col].cumsum() / np.arange(1, df.shape[0] + 1)
    new_col = '{}_full_history_diff'.format(col)
    df.loc[:, new_col] = df[col] - mean
    print(new_col)
    return new_col