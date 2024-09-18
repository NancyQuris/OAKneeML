import argparse
import datetime
import os
import sys
import random
import numpy as np
import pandas as pd 
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataset
sys.path.append('../')
import knee_net
import util_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train using image, image and clinical data, or only use clinical data with neural network.')
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'image_and_clinical', 'clinical'])
    parser.add_argument('--keys', type=list, default=util_fn.keys_of_interest)
    parser.add_argument('--label', type=str)
    parser.add_argument('--mcid', type=float)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--balanced', action='store_true')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_features', type=int, default=60)

    parser.add_argument('--net', type=str, default='convnext_tiny', choices=['resnet18', 'resnet101', 'resnet152', 'alexnet', 'convnext_tiny', 'convnext_base', 'vit_b_32'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    date_obj = datetime_obj.date()
    time_obj = datetime_obj.time()
    output_dir = str(date_obj.year) + 'y' + str(date_obj.month) + 'm' + str(date_obj.day) + \
        'd' + str(time_obj.hour) + 'h' + str(time_obj.minute) + 'm'

    os.mkdir(os.path.join('../../models/{}_{}_{}_mcid'.format(args.output_dir+output_dir, args.mode, args.label)))
    sys.stdout = util_fn.Tee(os.path.join('../../models/{}_{}_{}_mcid'.format(args.output_dir+output_dir, args.mode, args.label), 'output.txt'))
    sys.stderr = util_fn.Tee(os.path.join('../../models/{}_{}_{}_mcid'.format(args.output_dir+output_dir, args.mode, args.label), 'err.txt'))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.train_file))

    # Create writer for TensorBoard
    writer = SummaryWriter('runs/training_result_1')

    # Tracking average stats for each epoch
    tmp_run_loss = [0 for i in range(args.num_epoch)]
    
    tmp_run_auc = [0 for i in range(args.num_epoch)]
    tmp_run_f1 = [0 for i in range(args.num_epoch)]
    tmp_run_precision = [0 for i in range(args.num_epoch)]
    
    tmp_val_auc = [0 for i in range(args.num_epoch)]
    tmp_val_f1 = [0 for i in range(args.num_epoch)]
    tmp_val_precision = [0 for i in range(args.num_epoch)]
    
    tmp_tst_auc = [0 for i in range(args.num_epoch)]
    tmp_tst_f1 = [0 for i in range(args.num_epoch)]
    tmp_tst_precision = [0 for i in range(args.num_epoch)]
    
    k_folds = 5
    for fold in range(k_folds):
        # get a preprocessor 
        train_frame, val_frame = util_fn.get_train_val_data(train_df, fold+1)
        preprocessor = util_fn.encode_category_columns(util_fn.categorical_cols, util_fn.get_feature(train_frame, args.keys))
        
        train_set, val_set = get_dataset(args.mode, args.label, args.mcid, args.keys, train=True, val_fold=fold+1, preprocessor=preprocessor)
        print('\n------ training fold {} ------------'.format(fold+1))
        print('train num {}, val num {}'.format(len(train_set), len(val_set)))
        test_set = get_dataset(args.mode, args.label, args.mcid, args.keys, train=False, preprocessor=preprocessor)

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

        val_loader = data.DataLoader(val_set, batch_size=32, num_workers=8)
        test_loader = data.DataLoader(test_set, batch_size=32, num_workers=8)

        # Model Instantiation
        model = knee_net.KneeNet(args.mode, args.pretrained, args.num_features, \
                                            [1024, 512, 256, 128, 32], [0.001, 0.01, 0.01, 0.01, 0.01], 0.1, 2, args.net)
        model = model.to(device)
        
        # Creating Loss and Optimizer
  
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(image_loader), eta_min=0, last_epoch=-1)  
        
        message = '{0: <10} {1: <10} {2: <10} {3: <10} {4: <10} {5: <10} {6: <10} {7: <10} {8: <10} {9: <10} {10: <10}'.format('EPOCH', \
            'run loss', 'run auc', 'run f1', 'run precision', 'valid auc', 'valid f1', 'valid precision', 'test auc', 'test f1', 'test precision')
        print(message)
        

        for e in range(args.num_epoch):
            # Training
            model.train()
            running_loss = util_fn.image_train(args.mode, train_loader, model, loss_fn, optimizer, device)
            
            # Evaluation
            model.eval()
            run_auc, run_f1, run_precision = util_fn.image_test(args.mode, train_loader, model, device)
            valid_auc, valid_f1, valid_precision = util_fn.image_test(args.mode, val_loader, model, device)
            test_auc, test_f1, test_precision = util_fn.image_test(args.mode, test_loader, model, device)

            writer.add_scalars('Fold {} Results'.format(fold+1), {
                'Running Loss': running_loss,

                'Run AUC': run_auc,
                'Run F1': run_f1,
                'Run Precision': run_precision,

                'Validation AUC': valid_auc,
                'Validation F1': valid_f1,
                'Validation Precision': valid_precision,

                'Test AUC': test_auc,
                'Test F1': test_f1,
                'Test Precision': test_precision,
                }, e+1)
            
            # Storing average performance for each epoch
            tmp_run_loss[e] += running_loss

            tmp_run_auc[e] += run_auc
            tmp_run_f1[e] += run_f1
            tmp_run_precision[e] += run_precision

            tmp_val_auc[e] += valid_auc
            tmp_val_f1[e] += valid_f1
            tmp_val_precision[e] += valid_precision 

            tmp_tst_auc[e] += test_auc
            tmp_tst_f1[e] += test_f1
            tmp_tst_precision[e] += test_precision

            message = '{0: <10} {1: <10} {2: <10} {3: <10} {4: <10} {5: <10} {6: <10} {7: <10} {8: <10} {9: <10} {10: <10}'.format(e + 1, \
                    round(running_loss, 5), round(run_auc, 4), round(run_f1, 4), round(run_precision, 5), round(valid_auc, 4), \
                        round(valid_f1, 4), round(valid_precision, 5), round(test_auc, 4), round(test_f1, 4), round(test_precision, 5))
            print(message)

        # Save model 
        model_path = os.path.join('../../models/{}_{}_{}_mcid'.format(args.output_dir+output_dir, args.mode, args.label), 'F{}_network.pt'.format(fold+1))
        torch.save(
                {   
                    'fold': fold+1,
                    'model_state_dict': model.state_dict(),
                }, model_path)
        
    # Get average performance for each epoch across folds
    print('----------------------- final result -------------------------------')
    message = '{0: <10} {1: <10} {2: <10} {3: <10} {4: <10} {5: <10} {6: <10} {7: <10} {8: <10} {9: <10} {10: <10}'.format('EPOCH', \
            'run loss', 'run auc', 'run f1', 'run precision', 'valid auc', 'valid f1', 'valid precision', 'test auc', 'test f1', 'test precision')
    print(message)
    for e in range(args.num_epoch):
        writer.add_scalars('Epoch Average Results', {
                    'Running Loss': tmp_run_loss[e]/k_folds,

                    'Run AUC': tmp_run_auc[e]/k_folds,
                    'Run F1': tmp_run_f1[e]/k_folds,
                    'Run Precision': tmp_run_precision[e]/k_folds,

                    'Validation AUC': tmp_val_auc[e]/k_folds,
                    'Validation F1': tmp_val_f1[e]/k_folds,
                    'Validation Precision': tmp_val_precision[e]/k_folds,

                    'Test AUC': tmp_tst_auc[e]/k_folds,
                    'Test F1': tmp_tst_f1[e]/k_folds,
                    'Test Precision': tmp_tst_precision[e]/k_folds,
                    }, e+1)

        message = '{0: <10} {1: <10} {2: <10} {3: <10} {4: <10} {5: <10} {6: <10} {7: <10} {8: <10} {9: <10} {10: <10}'.format(\
                    e + 1, round(tmp_run_loss[e]/k_folds, 5), 
                    round(tmp_run_auc[e]/k_folds, 4), round(tmp_run_f1[e]/k_folds, 4), round(tmp_run_precision[e]/k_folds, 5), \
                    round(tmp_val_auc[e]/k_folds, 4), round(tmp_val_f1[e]/k_folds, 5), round(tmp_val_precision[e]/k_folds, 4),\
                    round(tmp_tst_auc[e]/k_folds, 4), round(tmp_tst_f1[e]/k_folds, 4), round(tmp_tst_precision[e]/k_folds, 5))
        print(message)
        
        
    