import argparse
import pandas as pd
import numpy as np

from features_config import selected_cols, droprows, std_reg_const, ridge_alpha
from ts_features import init_data, add_openclose_diff, add_diffs, add_shifts
from ts_features import add_intraday_ewma, add_rsi
from ts_features import add_time_depended_rolling, add_full_history_diff


def argparser():
    parser = argparse.ArgumentParser(description='Feature extractor')
    parser.add_argument('train_data_path', type=str, help='train data path')
    parser.add_argument('test_data_path', type=str, nargs='?', default=None, help='test data path')
    return parser.parse_args()


def add_xprice_features(data):
    agg_col = 'xprice'
    print('----->Adding {}-features...'.format(agg_col))
    
    mean_cols = add_time_depended_rolling(data, 'xprice', [6,60,360,1410], np.mean, 'mean')
    for col in mean_cols:
        data[col] = data[agg_col] - data[col]

    emwa_cols = add_intraday_ewma(data, 'xprice', [24])
    for col in emwa_cols:
        data[col] = data[agg_col] - data[col]

    add_shifts(data, 'xprice', [1410])

    add_rsi(data, 'xprice_time_mean_6', [6])
    add_shifts(data, 'xprice_time_mean_360', [1410])

    add_intraday_ewma(data, 'xprice_time_mean_60', [60])
    
    
def add_yprice_features(data):
    agg_col = 'yprice'
    print('----->Adding {}-features...'.format(agg_col))
    
    add_full_history_diff(data, agg_col)

    mean_cols = add_time_depended_rolling(data, agg_col, [60,360,720], np.mean, 'mean')
    mean_cols += add_intraday_ewma(data, agg_col, [24, 60])

    for col in mean_cols:
        data[col] = data[agg_col] - data[col]

    std_cols = add_time_depended_rolling(data, agg_col, [360,720], np.std, 'std')
    for col in std_cols:
        data[col] = data[col].fillna(0) + std_reg_const
    data['yprice_time_zscore_360'] = data.yprice_time_mean_360 / data.yprice_time_std_360
    data['yprice_time_zscore_720'] = data.yprice_time_mean_720 / data.yprice_time_std_720
    print(['yprice_time_zscore_360', 'yprice_time_zscore_720'])

    data['yprice_ewma_difpair_60_24'] = data.yprice_dayly_ewma_60 - data.yprice_dayly_ewma_24
    print(['yprice_ewma_difpair_60_24'])

    add_shifts(data, agg_col, [1410])
    add_rsi(data, 'yprice_time_mean_60', [360])
    add_shifts(data, 'yprice_time_mean_360', [1410, 2820])
    add_shifts(data, 'yprice_time_mean_60', [60])
    add_intraday_ewma(data, 'yprice_time_mean_60', [24])

    
def add_log_features(data):
    print('----->Adding log-features...')
    
    data['xlog'] = data.xprice.apply(np.log1p)
    data['ylog'] = data.yprice.apply(np.log1p)
    print(['xlog'])
    
    add_intraday_ewma(data, 'xlog', [60])
    add_intraday_ewma(data, 'ylog', [360]);

    data['xlog_dayly_ewma_60'] = data['xlog'] - data['xlog_dayly_ewma_60']
    data['ylog_dayly_ewma_360'] = data['ylog'] - data['ylog_dayly_ewma_360']
    print(['xlog_dayly_ewma_60', 'ylog_dayly_ewma_360'])

    
def add_garmonic_features(data):
    agg_col = 'xy_garmonic'
    print('----->Adding {}-features...'.format(agg_col))
    
    add_time_depended_rolling(data, agg_col, [1410], np.std, 'std')
    data['xy_garmonic_time_std_1410'] = data['xy_garmonic_time_std_1410'].fillna(0) + std_reg_const
    
    emwa_cols = add_intraday_ewma(data, agg_col, [360, 720])
    for col in emwa_cols:
        data[col] = data[agg_col] - data[col]
    data['xy_garmonic_ewma_prodpair_720_360'] = data.xy_garmonic_dayly_ewma_720 * data.xy_garmonic_dayly_ewma_360
    data.drop(['xy_garmonic_dayly_ewma_360', 'xy_garmonic_dayly_ewma_720'], 1, inplace=True)
    print(['xy_garmonic_ewma_prodpair_720_360'])

def add_geom_features(data):
    agg_col = 'xy_geom'
    print('----->Adding {}-features...'.format(agg_col))

    mean_cols = add_time_depended_rolling(data, agg_col, [6, 60, 360, 720], np.mean, 'mean')
    for col in mean_cols:
        data[col] = data[agg_col] - data[col]

    add_shifts(data, 'xy_geom_time_mean_360', [120])
    add_intraday_ewma(data, 'xy_geom_time_mean_60', [6])
    add_intraday_ewma(data, 'xy_geom_time_mean_720', [120])
    
    
def add_relation_features(data):
    print('----->Adding xy_relation-features...')
    std_cols = add_time_depended_rolling(data, 'xy_relation', [360, 720], np.std, 'std');
    for col in std_cols:
        data[col] = data[col].fillna(0) + std_reg_const    
        
def add_square_features(data):
    agg_col = 'xy_square'
    print('----->Adding {}-features...'.format(agg_col))
    
    mean_cols = add_time_depended_rolling(data, agg_col, [60], np.mean, 'mean')
    std_cols = add_time_depended_rolling(data, agg_col, [60], np.std, 'std')

    for col in mean_cols:
        data[col] = data[agg_col] - data[col]
    for col in std_cols:
        data[col] = data[col].fillna(0) + std_reg_const

    data['xy_square_time_zscore_60'] = data['xy_square_time_mean_60'] / data['xy_square_time_std_60']
    data.drop(['xy_square_time_mean_60', 'xy_square_time_std_60'], 1, inplace=True)
    print(['xy_square_time_zscore_60'])

def add_spread_features(data):
    agg_col = 'yx_spread'
    print('----->Adding {}-features...'.format(agg_col))
    
    mean_cols = add_time_depended_rolling(data, agg_col, [60,720,1410], np.mean, 'mean')
    for col in mean_cols:
        data[col] = data[agg_col] - data[col]

    add_time_depended_rolling(data, agg_col, [1410], np.std, 'std')
    data.yx_spread_time_std_1410 = data.yx_spread_time_std_1410.fillna(0) + std_reg_const
    data['yx_spread_time_zscore_1410'] = data.yx_spread_time_mean_1410 / data.yx_spread_time_std_1410
    print(['yx_spread_time_zscore_1410'])
    
    add_shifts(data, 'yx_spread_time_mean_60', [360])
    add_shifts(data, 'yx_spread_time_mean_720', [120])

    emwa_cols = add_intraday_ewma(data, agg_col, [60, 360])
    for col in emwa_cols:
        data[col] = data[agg_col] - data[col]
    data['yx_spread_ewma_prodpair_360_60'] = data.yx_spread_dayly_ewma_360 * data.yx_spread_dayly_ewma_60
    data.drop(['yx_spread_dayly_ewma_60', 'yx_spread_dayly_ewma_360'], 1, inplace=True)
    print(['yx_spread_ewma_prodpair_360_60'])

def selected_features_extractor(train_data_path, test_data_path=None):
    data, ntrain = init_data(train_data_path, test_data_path)
    add_openclose_diff(data)
    add_xprice_features(data)
    add_yprice_features(data)
    add_log_features(data)
    add_garmonic_features(data)
    add_geom_features(data)
    add_relation_features(data)
    add_square_features(data)
    add_spread_features(data)
    usecols = selected_cols + ['returns', 'periods_before_closing']
    
    train = data[usecols].iloc[droprows:ntrain]
    test = data[usecols].iloc[ntrain:]
    return train, test

    
def main():
    args = argparser()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    
    train, test = extract_features(train_data_path, test_data_path)
    train.to_pickle('train.pkl')
    test.to_pickle('test.pkl')
    
if __name__ == "__main__":
    main()