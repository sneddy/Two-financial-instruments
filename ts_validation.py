import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from copy import copy

def time_split(data, valid_ratio, test_ratio):
    n_valid = max(1, int(data.shape[0] * valid_ratio))
    n_test = max(1, int(data.shape[0] * test_ratio))
    n_train = data.shape[0] - n_valid - n_test
    
    train = data.iloc[:n_train].reset_index(drop=True).copy()
    valid = data.iloc[n_train:-n_test].reset_index(drop=True).copy()
    test = data.iloc[-n_test:].reset_index(drop=True).copy()
    merged_test = valid.append(test).reset_index(drop=True)
    return train, valid, test


def validate_sklearn_model(model, data, base_cols, valid_ratio, test_ratio, droprows=0, 
                           verbose=True, only_valid=False):
    selected_cols = copy(base_cols)
    helper_cols = list(set(selected_cols + ['periods_before_closing', 'returns']))
    train, valid, test = time_split(data[helper_cols], valid_ratio, test_ratio)
    train.drop(np.arange(droprows), inplace=True)
    train.dropna(inplace=True)
    
    if verbose:
        print('Data shapes: ', train.shape, valid.shape, test.shape)

    metrics_dict = {}
    
    if valid_ratio!=0:
        model.fit(train[selected_cols], train.returns)
        y_valid_predicted = model.predict(valid[selected_cols])
        
        y_valid_predicted[valid.periods_before_closing == 0] = 0
        
        metrics_dict['valid_mse'] = mean_squared_error(y_valid_predicted, valid.returns)
        metrics_dict['valid_r2'] = r2_score(valid.returns, y_valid_predicted) * 100
        if verbose:
            print('\nValid MSE: \t\t {:.5}'.format(metrics_dict['valid_mse']))
            print('Valid R2 (x100): \t {:.5}'.format(metrics_dict['valid_r2']))
    
    if not only_valid:
        model.fit(train.append(valid)[selected_cols], train.append(valid).returns)
        y_test_predicted = model.predict(test[selected_cols])
        y_test_predicted[test.periods_before_closing == 0] = 0

        metrics_dict['test_mse'] = mean_squared_error(y_test_predicted, test.returns)
        metrics_dict['test_r2'] = r2_score(test.returns, y_test_predicted) * 100
        if verbose:
            print('\nTest MSE: \t\t {:.5}'.format(metrics_dict['test_mse']))
            print('Test R2 (x100): \t {:.5}'.format(metrics_dict['test_r2']))
    
#     metrics_dict['model'] = model
    return metrics_dict


def greedy_add_del_strategy(model, data, cols, valid_ratio, test_ratio, droprows=0, add_frequency=1):
    selected_cols = cols.copy()
    removed_cols = []
    current_step = 0
    
    current_score = -float('inf')
    
    while selected_cols:
        current_step += 1
        if current_step % add_frequency == 0:
            for col in removed_cols:
                current_cols = selected_cols + [col]
                current_metrics = validate_sklearn_model(
                    model, data, current_cols,
                    valid_ratio=valid_ratio, test_ratio=test_ratio, droprows=droprows,
                    verbose=False, only_valid=True
                )
                if current_metrics['valid_r2'] > current_score:
                    current_score = current_metrics['valid_r2']
                    selected_cols.append(col)
                    print('added {}: r2: {:.5}'.format(col, current_score))

        best_score_by_iter = -float('inf')
        worst_col = ''
        for col in selected_cols:
            current_cols = [c for c in selected_cols if c!=col]
            current_metrics = validate_sklearn_model(
                model, data, current_cols, 
                valid_ratio, test_ratio, droprows,
                verbose=False, only_valid=True
            )

            if current_metrics['valid_r2'] > best_score_by_iter:
                best_score_by_iter = current_metrics['valid_r2']
                worst_col = col
        if best_score_by_iter > current_score:
            current_score = best_score_by_iter
            print('removed {}: r2: {:.5}'.format(worst_col, best_score_by_iter))
            selected_cols.remove(worst_col)
            removed_cols.append(worst_col)
        else:
            return selected_cols
        
def greedy_add_strategy(model, data, selected_cols, additional_cols, valid_ratio, test_ratio, droprows=0):
    base_cols = selected_cols.copy()
    current_score = validate_sklearn_model(
        model, data, base_cols,
        valid_ratio, test_ratio, droprows,
        verbose=False, only_valid=True
    )['valid_r2']
    is_continue_search = True
    while is_continue_search:
        is_continue_search = False
        for col in additional_cols:
            current_cols = base_cols + [col]
            current_metrics = validate_sklearn_model(
                model, data, current_cols,
                valid_ratio, test_ratio, droprows,
                verbose=False, only_valid=True
            )
            if current_metrics['valid_r2'] > current_score:
                current_score = current_metrics['valid_r2']
                base_cols.append(col)
                additional_cols.remove(col)
                is_continue_search = True
                print('added {}: r2: {:.5}'.format(col, current_score))
        
    return base_cols