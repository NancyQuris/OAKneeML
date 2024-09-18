import argparse
import copy
import os
import datetime 
import pickle
import sys 
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

import util_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get bootstrap performance using clinical data only with XGB.')
    parser.add_argument('--keys', type=list, default=util_fn.keys_of_interest)
    parser.add_argument('--label', type=str)
    parser.add_argument('--mcid', type=float)
    parser.add_argument('--n_bootstrap', type=int, default=1001)
    parser.add_argument('--n_estimator', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--colsample_bytree', type=float)
    parser.add_argument('--balanced', action='store_true')
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    date_obj = datetime_obj.date()
    time_obj = datetime_obj.time()
    output_dir = str(date_obj.year) + 'y' + str(date_obj.month) + 'm' + str(date_obj.day) + \
        'd' + str(time_obj.hour) + 'h' + str(time_obj.minute) + 'm'

    os.mkdir(os.path.join('../../models_bs/clinical_{}_{}_mcid'.format(output_dir, args.label)))
    sys.stdout = util_fn.Tee(os.path.join('../../models_bs/clinical_{}_{}_mcid'.format(output_dir, args.label), 'output.txt'))
    sys.stderr = util_fn.Tee(os.path.join('../../models_bs/clinical_{}_{}_mcid'.format(output_dir, args.label), 'err.txt'))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    train_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.train_file))
    test_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.test_file))

    preprocessor = util_fn.encode_category_columns(util_fn.categorical_cols, util_fn.get_feature(train_df, args.keys))
    train_features = preprocessor.transform(util_fn.get_feature(train_df, args.keys))
    train_labels = util_fn.get_label(train_df, args.label, args.mcid)
    
    print('dissat (label 1): {}, sat (label 0): {}'.format(util_fn.count_dissatisfied_patients(train_labels),\
                                                            util_fn.count_satisfied_patients(train_labels)))

    # print(type(train_features), train_features.shape, type(train_labels), train_labels.shape)

    model = XGBClassifier(random_state=0,
                        n_estimators=args.n_estimator,
                        max_depth=args.max_depth,
                        learning_rate=args.lr,
                        subsample=args.subsample,
                        colsample_bytree=args.colsample_bytree,
                        eval_metric='auc')
    
    if args.balanced:
        weights = compute_sample_weight(class_weight='balanced', y=train_labels)
    else:
        weight = None

    model = model.fit(train_features, train_labels, sample_weight=weights)
    
    save_feature_column_name = os.path.join('../../models_bs/clinical_{}_{}_mcid'.format(output_dir, args.label), 'feature_name.txt')
    with open(save_feature_column_name, 'w') as f:
        print(preprocessor.get_feature_names_out(), file=f)
    f.close()

    saved_model_filename = os.path.join('../../models_bs/clinical_{}_{}_mcid'.format(output_dir, args.label), 'model.dat')
    pickle.dump(model, open(saved_model_filename, 'wb'))
    
    message = '\n{:<10} {:<10} {:<10} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13}'.format(
                'bs #', 'test #', 'patients #', 
                'auc', 'f1', 'precision', 'sensitivity', 'specificity', 
                'tn', 'fp', 'fn', 'tp')
    print(message)

    train_result = util_fn.test(model, train_features, train_labels)
    print('{:<10} {:<10} {:<10} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13}'.format(
        'train', train_features.shape[0], len(set(train_df['IC'].values)), \
        round(train_result[0], 4), round(train_result[1], 4), round(train_result[2], 4), 
        round(train_result[3], 4), round(train_result[4], 4), round(train_result[5], 4), 
        round(train_result[6], 4), round(train_result[7], 4), round(train_result[8], 4)))
    print('------------------------------------------------------------------------------------------------')

    test_aucs = []
    test_f1s = []
    test_specificities = []
    test_sensitivities = []
    test_precisions = []
    test_tns = []
    test_fps = []
    test_fns = []
    test_tps = []
    test_fprs = []
    test_tprs = []
    test_thresholds = []
    test_pres = []
    test_recas = []
    test_pr_thresholds = []

    patients = set(test_df['IC'].values)
    sat_patients = set(test_df.loc[test_df[args.label]>=args.mcid]['IC'].values)
    dissat_patients = set(test_df.loc[test_df[args.label]<args.mcid]['IC'].values)
    
    for i in range(args.n_bootstrap):
        # bootstrap at patient level 
        selected_patients = util_fn.get_sample_list(patients, sat_patients, dissat_patients)

        bs_df = pd.DataFrame(columns=test_df.columns)
        for p in selected_patients:
            row_to_add = test_df.loc[(test_df['IC'] == p)]
            bs_df = pd.concat([bs_df, copy.deepcopy(row_to_add)])

        test_features = preprocessor.transform(util_fn.get_feature(bs_df, args.keys))
        test_labels = util_fn.get_label(bs_df, args.label, args.mcid).astype('int32')

        test_auc, test_f1, test_precision, test_sensitivity, test_specificity, \
            test_tn, test_fp, test_fn, test_tp, test_fpr, test_tpr, test_threshold, test_pre, test_reca, test_pr_threshold = util_fn.test(model, test_features, test_labels, roc=True)

        message = '{:<10} {:<10} {:<10} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13}'.format(\
                i+1, len(bs_df), len(set(selected_patients)), round(test_auc, 4), round(test_f1, 4),\
                round(test_precision, 4), round(test_sensitivity, 4), round(test_specificity, 4), \
                round(test_tn, 4), round(test_fp, 4), round(test_fn, 4), round(test_tp, 4))
        print(message)

        test_aucs.append(test_auc)
        test_f1s.append(test_f1)
        test_precisions.append(test_precision)
        test_sensitivities.append(test_sensitivity)
        test_specificities.append(test_specificity)
        test_tns.append(test_tn)
        test_fps.append(test_fp)
        test_fns.append(test_fn)
        test_tps.append(test_tp)
        test_fprs.append(test_fpr)
        test_tprs.append(test_tpr)
        test_thresholds.append(test_threshold)
        test_pres.append(test_pre)
        test_recas.append(test_reca)
        test_pr_thresholds.append(test_pr_threshold)

    test_aucs.sort()
    test_f1s.sort()
    test_precisions.sort()
    test_sensitivities.sort()
    test_specificities.sort()
    test_tns.sort()
    test_fps.sort()
    test_fns.sort()
    test_tps.sort()

    low_percent = int(args.n_bootstrap * 0.025)
    high_percent = int(args.n_bootstrap * 0.975)  

    print('test auc mean {}, CI {}-{}'.format(\
        round(sum(test_aucs)/args.n_bootstrap, 4), round(test_aucs[low_percent], 4), round(test_aucs[high_percent], 4)))
    print('test f1 mean {}, CI {}-{}'.format(\
        round(sum(test_f1s)/args.n_bootstrap, 4) , round(test_f1s[low_percent], 4), round(test_f1s[high_percent], 4)))
    print('test precision mean {}, CI {}-{}'.format(\
        round(sum(test_precisions)/args.n_bootstrap, 4), round(test_precisions[low_percent], 4), round(test_precisions[high_percent], 4)))
    print('test sensitivity mean {}, CI {}-{}'.format(\
        round(sum(test_sensitivities)/args.n_bootstrap, 4), round(test_sensitivities[low_percent], 4), round(test_sensitivities[high_percent], 4)))
    print('test specificity mean {}, CI {}-{}'.format(\
        round(sum(test_specificities)/args.n_bootstrap, 4), round(test_specificities[low_percent], 4), round(test_specificities[high_percent], 4)))
    print('test tn mean {}, CI {}-{}'.format(\
        round(sum(test_tns)/args.n_bootstrap, 4), round(test_tns[low_percent], 4), round(test_tns[high_percent], 4)))
    print('test fp mean {}, CI {}-{}'.format(\
        round(sum(test_fps)/args.n_bootstrap, 4), round(test_fps[low_percent], 4), round(test_fps[high_percent], 4)))
    print('test fn mean {}, CI {}-{}'.format(\
        round(sum(test_fns)/args.n_bootstrap, 4), round(test_fns[low_percent], 4), round(test_fns[high_percent], 4)))
    print('test tp mean {}, CI {}-{}'.format(\
        round(sum(test_tps)/args.n_bootstrap, 4), round(test_tps[low_percent], 4), round(test_tps[high_percent], 4)))
    
    # https://towardsdatascience.com/pooled-roc-with-xgboost-and-plotly-553a8169680c
    fpr_mean = np.linspace(0, 1, args.n_bootstrap)
    interp_tprs = []
    for i in range(args.n_bootstrap):
        fpr = test_fprs[i]
        tpr = test_tprs[i]
        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    
    with open(os.path.join('../../models_bs/clinical_{}_{}_mcid'.format(output_dir, args.label), 'roc.txt'), 'w') as f:
        f.write('{:<10} {:<10}\n'.format('fpr', 'tpr'))
        for i in range(args.n_bootstrap):
            f.write('{:<10} {:<10}\n'.format(round(fpr_mean[i], 6), round(tpr_mean[i], 6)))
    f.close()
    
    