droprows = 7050
std_reg_const = 0.1
normalization_std_reg = 0.0001
ridge_alpha = 50


selected_cols = [
    'is_end_of_week',
    'weekday',
    'xdiff_from_closing',
    'xdiff_from_opening',
    'xlog',
    'xlog_dayly_ewma_60',
    'xprice_dayly_ewma_24',
    'xprice_lag_1410',
    'xprice_time_mean_1410',
    'xprice_time_mean_360_lag_1410',
    'xprice_time_mean_6',
    'xprice_time_mean_60_dayly_ewma_60',
    'xprice_time_mean_6_rsi_6',
    'xy_garmonic_ewma_prodpair_720_360',
    'xy_garmonic_time_std_1410',
    'xy_geom_time_mean_360_lag_120',
    'xy_geom_time_mean_6',
    'xy_geom_time_mean_60_dayly_ewma_6',
    'xy_geom_time_mean_720_dayly_ewma_120',
    'xy_relation_time_std_360',
    'xy_relation_time_std_720',
    'xy_square_time_zscore_60',
    'ydiff_from_closing',
    'ylog_dayly_ewma_360',
    'yprice_dayly_ewma_60',
    'yprice_lag_1410',
    'yprice_ewma_difpair_60_24',
    'yprice_full_history_diff',
    'yprice_time_mean_360',
    'yprice_time_mean_360_lag_1410',
    'yprice_time_mean_360_lag_2820',
    'yprice_time_mean_60',
    'yprice_time_mean_60_dayly_ewma_24',
    'yprice_time_mean_60_lag_60',
    'yprice_time_mean_60_rsi_360',#?
    'yprice_time_mean_720',
    'yprice_time_zscore_360',
    'yprice_time_zscore_720',
    'yx_spread_ewma_prodpair_360_60',
    'yx_spread_time_mean_60_lag_360',
    'yx_spread_time_mean_720_lag_120',
    'yx_spread_time_zscore_1410',
    
] 