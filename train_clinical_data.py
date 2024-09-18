import argparse
import os
import datetime 
import sys 

import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

import util_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train using clinical data only with XGB.')
    parser.add_argument('--keys', type=list, default=util_fn.keys_of_interest)
    parser.add_argument('--label', type=str)
    parser.add_argument('--mcid', type=float)
    parser.add_argument('--balanced', action='store_true')
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    date_obj = datetime_obj.date()
    time_obj = datetime_obj.time()
    output_dir = str(date_obj.year) + 'y' + str(date_obj.month) + 'm' + str(date_obj.day) + \
        'd' + str(time_obj.hour) + 'h' + str(time_obj.minute) + 'm'

    os.mkdir(os.path.join('../../models/clinical_{}_{}_mcid'.format(output_dir, args.label)))
    image_model_path = '../../models/clinical_{}_{}_mcid'.format(output_dir, args.label)
    sys.stdout = util_fn.Tee(os.path.join('../../models/clinical_{}_{}_mcid'.format(output_dir, args.label), 'output.txt'))
    sys.stderr = util_fn.Tee(os.path.join('../../models/clinical_{}_{}_mcid'.format(output_dir, args.label), 'err.txt'))

    train_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.train_file))
    test_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.test_file))

    num_folds = 5 

    for fold in range(num_folds):
        train_frame, val_frame = util_fn.get_train_val_data(train_df, fold+1)
        train_labels = util_fn.get_label(train_frame, args.label, args.mcid)
        val_labels = util_fn.get_label(val_frame, args.label, args.mcid) 

        print('-- fold {} --'.format(fold+1))
        print('train num {}, num of sat patients {}, num of dissat patients {}'.format(
            len(train_labels), util_fn.count_satisfied_patients(train_labels), util_fn.count_dissatisfied_patients(train_labels))) 
        print('val num {}, num of sat patients {}, num of dissat patients {}'.format(
            len(val_labels), util_fn.count_satisfied_patients(val_labels), util_fn.count_dissatisfied_patients(val_labels))) 
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.2, 0.3],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    parameter_search_list = list(ParameterGrid(param_grid))

    for p in parameter_search_list:
        print('\n#############################################')
        
        total_val_precision = 0
        total_val_auc = 0
        message = '{0: <10} {1: <10} {2: <10} {3: <10} {4: <10} {5: <10} {6: <10} {7: <10} {8: <10} {9:<10}'.format(\
                'fold', 'run auc', 'run f1', 'run precision', 'valid auc', 'valid f1', 'valid precision', 'test auc', 'test f1', 'test precision')
        print(message)
        for fold in range(num_folds):
            model = XGBClassifier(random_state=0,
                                  n_estimators=p['n_estimators'],
                                  max_depth=p['max_depth'],
                                  learning_rate=p['learning_rate'],
                                  subsample=p['subsample'],
                                  colsample_bytree=p['colsample_bytree'],
                                  eval_metric='auc'
                                  )

            train_frame, val_frame = util_fn.get_train_val_data(train_df, fold+1)
            preprocessor = util_fn.encode_category_columns(util_fn.categorical_cols, util_fn.get_feature(train_frame, args.keys))
            train_features = preprocessor.transform(util_fn.get_feature(train_frame, args.keys))
            train_labels = util_fn.get_label(train_frame, args.label, args.mcid)
            if args.balanced:
                weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            else:
                weights = None 
            
            model = model.fit(train_features, train_labels, sample_weight=weights)
            
            val_features = preprocessor.transform(util_fn.get_feature(val_frame, args.keys))
            val_labels = util_fn.get_label(val_frame, args.label, args.mcid) 
            
            test_features = preprocessor.transform(util_fn.get_feature(test_df, args.keys))
            test_labels = util_fn.get_label(test_df, args.label, args.mcid)

            run_result = util_fn.test(model, train_features, train_labels)
            run_auc, run_f1, run_precision = run_result[0], run_result[1], run_result[2]
            valid_result = util_fn.test(model, val_features, val_labels)
            valid_auc, valid_f1, valid_precision = valid_result[0], valid_result[1], valid_result[2]
            test_result = util_fn.test(model, test_features, test_labels)
            test_auc, test_f1, test_precision = test_result[0], test_result[1], test_result[2]


            # add test and f1, check how grid search is conducted 
            message = '{0: <10} {1: <10} {2: <10} {3: <10} {4: <10} {5: <10} {6: <10} {7: <10} {8: <10} {9:<10}'.format(\
                fold+1, round(run_auc, 5), round(run_f1, 4), round(run_precision, 4), round(valid_auc, 5), round(valid_f1, 4), \
                round(valid_precision, 4), round(test_auc, 5), round(test_f1, 4), round(test_precision, 4))
            print(message)

            total_val_precision += valid_precision
            total_val_auc += valid_auc

        print('---------------------------------------------')
        print('n_estimator {}, max_depth {}, learning_rate {}, subsample {}, colsample_by_tree {}'.format(
                p['n_estimators'],
                p['max_depth'],
                p['learning_rate'],
                p['subsample'],
                p['colsample_bytree']
            )
              )
        print('valid precision result: {}'.format(total_val_precision/num_folds))
        print('valid auc result: {}'.format(total_val_auc/num_folds))



    
    

