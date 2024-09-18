import argparse
import copy 
import datetime
import os
import sys
import random
import numpy as np
import pandas as pd 
import torch
import torch.utils.data as data

from dataset import KneeDataset
import knee_net
import util_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get bootstrap performance of image, image and clinical data, or only use clinical data with neural network.')
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'image_and_clinical', 'clinical'])
    parser.add_argument('--keys', type=list, default=util_fn.keys_of_interest)
    parser.add_argument('--label', type=str)
    parser.add_argument('--mcid', type=float)
    parser.add_argument('--n_bootstrap', type=int, default=1001)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--balanced', action='store_true')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_features', type=int, default=60)

    parser.add_argument('--net', type=str, default='convnext_tiny', choices=['resnet18', 'resnet101', 'resnet152', 'alexnet', 'convnext_tiny', 'convnext_base', 'vit_b_32'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()

    datetime_obj = datetime.datetime.now()
    date_obj = datetime_obj.date()
    time_obj = datetime_obj.time()
    output_dir = str(date_obj.year) + 'y' + str(date_obj.month) + 'm' + str(date_obj.day) + \
        'd' + str(time_obj.hour) + 'h' + str(time_obj.minute) + 'm'

    
    image_model_path = '../../models_bs/{}_{}_{}_mcid'.format(args.output_dir+output_dir, args.mode, args.label)
    os.mkdir(image_model_path)
    sys.stdout = util_fn.Tee(os.path.join(image_model_path, 'output.txt'))
    sys.stderr = util_fn.Tee(os.path.join(image_model_path, 'err.txt'))

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
    preprocessor = util_fn.encode_category_columns(util_fn.categorical_cols, util_fn.get_feature(train_df, args.keys))
    train_dataset = KneeDataset(train_df, args.mode, args.label, args.mcid, args.keys, transform=True, preprocessor=preprocessor)

    save_feature_column_name = os.path.join('../../models_bs/{}_{}_{}_mcid'.format(args.output_dir+output_dir, args.mode, args.label), 'feature_name.txt')
    with open(save_feature_column_name, 'w') as f:
        print(preprocessor.get_feature_names_out(), file=f)
    f.close()

    if args.balanced:
        weights = util_fn.make_weights_for_balanced_classes(train_dataset, 2)
        weights = torch.DoubleTensor(weights)  
        if len(weights) > 0:
            sampler = data.sampler.WeightedRandomSampler(weights, len(weights))
        else:
            print("Error: No weights available. Please check the implementation of make_weights_for_balanced_classes.")
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8)
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)

    # Model Instantiation
    model = knee_net.KneeNet(args.mode, args.pretrained, args.num_features, \
                                        [1024, 512, 256, 128, 32], [0.001, 0.01, 0.01, 0.01, 0.01], 0.1, 2, args.net)
    model = model.to(device)

    # Creating Loss and Optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(image_loader), eta_min=0, last_epoch=-1)
    
        
    # train loop to get model 
    for e in range(args.num_epoch):
        # Training
        model.train()
        running_loss = util_fn.image_train(args.mode, train_loader, model, loss_fn, optimizer, device)
    
    # save model
    model_path = os.path.join(image_model_path, 'network.pt')
    torch.save(model.state_dict(), model_path)

    model.eval()
    run_auc, run_f1, run_precision = util_fn.image_test(args.mode, train_loader, model, device)
    print('run auc: {0:<6}, run f1: {1:<6}, run_precision: {2:<6}'.format(\
        round(run_auc, 4), round(run_f1, 4), round(run_precision, 4)))
    
    # bootstrap 
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

    message = '\n{:<10} {:<10} {:<10} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13} {:<13}'.format(
                'bs #', 'test #', 'patients #', 
                'auc', 'f1', 'precision', 'sensitivity', 'specificity', 
                'tn', 'fp', 'fn', 'tp')
    print(message)

    test_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.test_file))
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

        test_set = KneeDataset(bs_df, args.mode, args.label, args.mcid, args.keys, transform=False, preprocessor=preprocessor)
        test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=8)

        test_auc, test_f1, test_precision, test_sensitivity, test_specificity, \
            test_tn, test_fp, test_fn, test_tp, test_fpr, test_tpr, test_threshold, test_pre, test_reca, test_pr_threshold = util_fn.image_bs_test(args.mode, test_dataloader, model, device, roc=True)

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
    
    with open(os.path.join(image_model_path, 'roc.txt'), 'w') as f:
        f.write('{:<10} {:<10}\n'.format('fpr', 'tpr'))
        for i in range(args.n_bootstrap):
            f.write('{:<10} {:<10}\n'.format(round(fpr_mean[i], 6), round(tpr_mean[i], 6)))
    f.close()


