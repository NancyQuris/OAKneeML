import argparse
import os
import datetime 
import sys 
import random 

import numpy as np 
import torch 
import torch.utils.data as data
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

from dataset import get_xgb_multimodal_dataset, get_dataset
import knee_net
import util_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train using image and clinical data with XGB.')
    parser.add_argument('--keys', type=list, default=util_fn.keys_of_interest)
    parser.add_argument('--label', type=str)
    parser.add_argument('--mcid', type=float)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--net', type=str, default='convnext_tiny', choices=['resnet18', 'resnet101', 'resnet152', 'alexnet', 'convnext_tiny', 'convnext_base', 'vit_b_32'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--num_features', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    date_obj = datetime_obj.date()
    time_obj = datetime_obj.time()
    if args.output_dir is None:
        output_dir = str(date_obj.year) + 'y' + str(date_obj.month) + 'm' + str(date_obj.day) + \
            'd' + str(time_obj.hour) + 'h' + str(time_obj.minute) + 'm'
    else:
        output_dir = args.output_dir

    os.mkdir(os.path.join('../../models/multimodal_xgb_{}_{}_mcid'.format(output_dir, args.label)))
    image_model_path = '../../models/multimodal_xgb_{}_{}_mcid'.format(output_dir, args.label)
    sys.stdout = util_fn.Tee(os.path.join('../../models/multimodal_xgb_{}_{}_mcid'.format(output_dir, args.label), 'output.txt'))
    sys.stderr = util_fn.Tee(os.path.join('../../models/multimodal_xgb_{}_{}_mcid'.format(output_dir, args.label), 'err.txt'))

    train_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.train_file))
    test_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.test_file))

    num_folds = 5 

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # get basic information of each fold 
    for fold in range(num_folds):
        train_frame, val_frame = util_fn.get_train_val_data(train_df, fold+1)
        train_labels = util_fn.get_label(train_frame, args.label, args.mcid)
        val_labels = util_fn.get_label(val_frame, args.label, args.mcid) 

        print('-- fold {} --'.format(fold+1))
        print('train num {}, num of sat patients {}, num of dissat patients {}'.format(
            len(train_labels), util_fn.count_satisfied_patients(train_labels), util_fn.count_dissatisfied_patients(train_labels))) 
        print('val num {}, num of sat patients {}, num of dissat patients {}'.format(
            len(val_labels), util_fn.count_satisfied_patients(val_labels), util_fn.count_dissatisfied_patients(val_labels))) 

    # train multimodal neural network to learn image features that are compliment to clinical data 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for fold in range(num_folds):
        train_frame, val_frame = util_fn.get_train_val_data(train_df, fold+1)
        preprocessor = util_fn.encode_category_columns(util_fn.categorical_cols, util_fn.get_feature(train_frame, args.keys))
        
        train_set, val_set = get_dataset('image_and_tabular', args.label, args.mcid, args.keys, train=True, val_fold=fold+1, preprocessor=preprocessor)
        print('\n------ training image model for fold {} ------------'.format(fold+1))
        
        if args.balanced:
            weights = util_fn.make_weights_for_balanced_classes(train_set, 2)
            weights = torch.DoubleTensor(weights)  
            if len(weights) > 0:
                sampler = data.sampler.WeightedRandomSampler(weights, len(weights))
            else:
                print("Error: No weights available. Please check the implementation of make_weights_for_balanced_classes.")

            train_loader = data.DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=8)
        else:
            train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=8)

        # Model Instantiation
        model = knee_net.KneeNet('image_and_tabular', args.pretrained, args.num_features, \
                                            [1024, 512, 256, 128, 32], [0.001, 0.01, 0.01, 0.01, 0.01], 0.1, 2, args.net)
        model = model.to(device)
        
        # Creating Loss and Optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for e in range(args.num_epoch):
            # Training
            model.train()
            running_loss = util_fn.image_train('image_and_tabular', train_loader, model, loss_fn, optimizer, device)

        model_path = os.path.join(image_model_path, f'{fold}.pt')
        torch.save(model.state_dict(), model_path)

    
    # train the xgb model 
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 10, 30],
        'learning_rate': [0.1, 0.01, 0.001],
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
             
            # load multimodal neural network
            state_dict = torch.load(os.path.join(image_model_path, f'{fold}.pt'), map_location='cpu')
            # only keep parameters of image featurizer  
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('image_net.model.'):
                    new_key = k[16:]
                    new_state_dict[new_key] = v
            image_model = knee_net.KneeNet('image', False, len(args.keys), [1024, 512, 256, 128, 32], [0.001, 0.01, 0.01, 0.01, 0.01], 0.1, 2, args.net)
            image_model.image_net.model.load_state_dict(new_state_dict)
            image_model.eval()
            

            train_set, val_set = get_xgb_multimodal_dataset('image_and_tabular', args.label, args.mcid, args.keys, train=True, val_fold=fold+1, preprocessor=preprocessor)
            test_set = get_xgb_multimodal_dataset('image_and_tabular', args.label, args.mcid, args.keys, train=False, preprocessor=preprocessor)
            
            train_loader = data.DataLoader(train_set, batch_size=32, num_workers=8, shuffle=False)
            val_loader = data.DataLoader(val_set, batch_size=32, num_workers=8)
            test_loader = data.DataLoader(test_set, batch_size=32, num_workers=8)

            # iterate through data loader to extract image embedding, tabular data and label
            train_features, train_labels = util_fn.get_feature_and_label_xgb(train_loader, image_model)

            if args.balanced:
                weights = compute_sample_weight(class_weight='balanced', y=train_labels)
            else:
                weights = None 

            # train the XGB model 
            model = model.fit(train_features, train_labels, sample_weight=weights)
            
            # test model 
            val_features, val_labels = util_fn.get_feature_and_label_xgb(val_loader, image_model) 
            test_features, test_labels = util_fn.get_feature_and_label_xgb(test_loader, image_model)

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