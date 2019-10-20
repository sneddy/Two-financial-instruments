import argparse
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge

from features_config import selected_cols, droprows, std_reg_const, normalization_std_reg, ridge_alpha
from feature_extractor import selected_features_extractor
from ts_validation import validate_model_by_pentate
from helper import print_importances


def argparser():
    parser = argparse.ArgumentParser(description='Fitting model')
    parser.add_argument('train_data_path', type=str, help='train data path')
    parser.add_argument(
        'path_store_model', type=str, nargs='?', default='ridge_weights.model',
        help='path to the trained model'
    )
    return parser.parse_args()

def normalize_train(df):
    extended_cols = selected_cols + ['returns', 'periods_before_closing']
    norm_train = df[extended_cols].reset_index(drop=True).copy()
    norm_mean = norm_train[selected_cols].mean()
    norm_std = norm_train[selected_cols].std() + normalization_std_reg
    norm_train.loc[:,selected_cols] = (norm_train[selected_cols] - norm_mean) / norm_std
    return norm_train

def main():
    print('--------->Start Estimation....')
    args = argparser()
    train_data_path = args.train_data_path
    store_model_path = args.path_store_model
    
    train, _ = selected_features_extractor(train_data_path)
    
    print('----->Normalization....')
    norm_train = normalize_train(train)
    model = Ridge(alpha=ridge_alpha)
    
    print('----->Validation....')
    print(validate_model_by_pentate(model, norm_train, selected_cols, 0))
    
    print('----->Fitting....')
    model = Ridge(alpha=ridge_alpha)
    model.fit(norm_train[selected_cols], norm_train.returns)
    
    print('----->Calculation feature importances...')
    print_importances(model, selected_cols)
    
    print('----->Saving model....')
    with open(store_model_path, 'wb') as file:
        pickle.dump(model, file)
    
    
if __name__ == "__main__":
    main()