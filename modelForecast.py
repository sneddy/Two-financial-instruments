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
        'test_data_path', type=str, nargs='?', default=None, 
        help='test data path'
    )
    parser.add_argument(
        'path_store_model', type=str, nargs='?', default='ridge_weights.model',
        help='path to the place where the model will be saved'
    )
    parser.add_argument(
        'path_to_test_predicted', type=str, nargs='?', default='predictions',
        help='path to the place where the predictions will be saved'
    )
    return parser.parse_args()

def normalize_train_test(train, test):
    train_extended_cols = selected_cols + ['returns', 'periods_before_closing']
    test_extended_cols = selected_cols + ['periods_before_closing']
    
    norm_train = train[train_extended_cols].reset_index(drop=True).copy()
    norm_test = test[test_extended_cols].reset_index(drop=True).copy()
    
    norm_mean = norm_train[selected_cols].mean()
    norm_std = norm_train[selected_cols].std() + normalization_std_reg
    
    norm_train.loc[:,selected_cols] = (norm_train[selected_cols] - norm_mean) / norm_std
    norm_test.loc[:,selected_cols] = (norm_test[selected_cols] - norm_mean) / norm_std
    return norm_train, norm_test

def main():
    print('--------->Starting Forecasting....')
    args = argparser()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    store_model_path = args.path_store_model
    predictions_path = args.path_to_test_predicted
    
    print('----->Loading trained model....')
    with open(store_model_path, 'rb') as file:
        trained_model = pickle.load(file)
        
    train, test = selected_features_extractor(train_data_path, test_data_path)
    
    print('----->Normalization....')
    _, norm_test = normalize_train_test(train, test)
  
    print('----->Prediction....')
    predicted = trained_model.predict(norm_test[selected_cols])
    
    print('----->Saving predictions....')
    with open(store_model_path, 'wb') as file:
        np.save(predictions_path, predicted)
    
    
if __name__ == "__main__":
    main()