import copy
import random 
import sys 
import torch 

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve


train_file = '../preprocess/train.csv'
test_file = '../preprocess/test.csv'

knee_MCID = 34.5
oxford_MCID = 5
PCS_MCID = 10
MCS_MCID = 10
Exp_threshold = 4
Sat_threshold = 4

keys_of_interest = ['AGE', 'GENDER', 'BMI_group', \
                    'RACE', 'LANG', 'EDU', \
                    'HT_0', 'WT_0',  \
                    'PREOP.PC', 'PREOP.roms', 'PREOP.rome', \
                    'PREOP.function', 'PREOP.knee', 'PREOP.oxford', \
                    'PREOP.Q1', 'PREOP.Q3', 'PREOP.Q5', 'PREOP.Q7', 'PREOP.Q12', \
                    'PREOP.sf36 1', 'PREOP.sf36 2', 'PREOP.sf36 3', 'PREOP.sf36 4', 'PREOP.sf36 5', \
                    'PREOP.sf36 6', 'PREOP.sf36 7', 'PREOP.sf36 8',\
                    'PREOP.PCS', 'PREOP.MCS', 'Diabetes', 'IHD', 'Stroke', \
                    'Arthritis', 'Asthma', 'Depression', 'Hypertension', 'Hyperlipidemia'] 

categorical_cols = ['BMI_group', 'RACE', 'LANG', 'EDU', 'GENDER', 'Diabetes', 'IHD', 'Stroke', \
                        'Arthritis', 'Asthma', 'Depression', 'Hypertension', 'Hyperlipidemia']


def preprocess_dataframe(data):
    data['RACE'] = data['RACE'].map({1: 'Chinese', 2: 'Malay', 3: 'Indian', 4: 'Others'})
    data['LANG'] = data['LANG'].map({1: 'English', 2: 'Mandarin', 3: 'Hokkien', 4: 'Cantonese', 5: 'Malay', 6: 'Others'})
    data['EDU'] = data['EDU'].map({1: 'none', 2: 'primary', 3: 'secondary', 4: 'tertiary'})
    data['BMI_group'] = pd.cut(data['BMI_0'], bins=[0, 23.9, 27.9, 31.9, 100], labels=['Normal', 'Overweight', 'Obese', 'Severely Obese'])
    data["GENDER"].replace({"FEMALE": 0, "MALE": 1}, inplace=True)
    return data 


def get_feature(data, key):
    return data[key]

def get_label(data, label, mcid): 
    target = data[label].values
    results = np.where(target<mcid, 1, 0)

    return results


def get_feature_and_label_xgb(data_loader, image_net):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_net.to(device)
    
    features, labels = [], []
    with torch.no_grad():
        for counter, (x, y) in enumerate(data_loader):
            image = x['image'].to(device)
            record = x['tabular_info']

            image_embedding = image_net(image).cpu()
            current_feature = torch.cat((image_embedding, record), dim=1)

            features.append(current_feature)
            labels.append(y)
            
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features.numpy(), labels.numpy()


def get_train_val_data(df, val_fold):
    train_df = df.loc[df['fold']!=val_fold]
    val_df = df.loc[df['fold']==val_fold]
    return copy.deepcopy(train_df), copy.deepcopy(val_df)


def encode_category_columns(categorical_cols, features):
    numeric_cols = [item for item in features.columns if item not in categorical_cols]
    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numeric_cols),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Fit the preprocessing pipeline on the training data
    features = preprocessor.fit(features)
    return preprocessor


def count_satisfied_patients(y):
    return sum(y == 0)


def count_dissatisfied_patients(y):
    return sum(y == 1)


def get_sample_list(patient_list, sat_patients, dissat_patients):
    sample_list = random.choices(list(patient_list), k=len(patient_list))
    if bool(set(sample_list) & sat_patients) and bool(set(sample_list) & dissat_patients):
        return sample_list
    else:
        return get_sample_list(patient_list, sat_patients, dissat_patients)


def test(model, x, y, roc=False):
    pred_y_prob = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, pred_y_prob)
        
    pred_y = model.predict(x)
    f1 = f1_score(y, pred_y)
    precision = precision_score(y, pred_y)
    recall = recall_score(y, pred_y)
    tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()
    specificity = tn / (tn+fp)

    if roc:
        fpr, tpr, threshold = roc_curve(y, pred_y_prob)
        pre, reca, pr_threshold = precision_recall_curve(y, pred_y_prob)
        return auc, f1, precision, recall, specificity, tn, fp, fn, tp, fpr, tpr, threshold, pre, reca, pr_threshold
    else:
        return auc, f1, precision, recall, specificity, tn, fp, fn, tp


def image_train(mode, loader, model, loss_fn, optimizer, device):
    running_loss = 0.
    for counter, (x, y) in enumerate(loader):
        if mode == 'image':
            x = x.to(device)
            y = y.to(device)
            output = model(image=x)
            
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif mode == 'clinical':
            x = x.to(device)
            y = y.to(device)
            output = model(tab=x)
            
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif mode == 'image_and_clinical':
            image = x['image']
            record = x['tabular_info']
            
            image = image.to(device)
            record = record.to(device)
            y = y.to(device)
            output = model(image=image, tab=record)

            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            raise NotImplementedError

        running_loss += loss.item()
        
    running_loss = running_loss/counter
    return running_loss

def image_test(mode, loader, model, device):
    y_true = []
    y_pred = []
    predicted_prob = []
    
    for x, y in loader:
        # added line to prevent memory overflow
        with torch.no_grad():
            if mode == 'image':
                x = x.to(device)
                y = y.to(device)
                output = model(image=x)
            elif mode == 'clinical':
                x = x.to(device)
                y = y.to(device)
                output = model(tab=x)
            elif mode == 'image_and_clinical':
                image = x['image']
                record = x['tabular_info']
            else:
                raise NotImplementedError

                image = image.to(device)
                record = record.to(device)
                y = y.to(device)
                output = model(image=image, tab=record)

        y_true.append(y.cpu())
        y_pred.append(output.argmax(1).cpu())
        predicted_prob.append(torch.softmax(output, dim=1).cpu())
        
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    predicted_prob = torch.cat(predicted_prob).detach().numpy()
    
    return roc_auc_score(y_true, predicted_prob[:, 1]), f1_score(y_true, y_pred), precision_score(y_true, y_pred)

def image_bootstrap_test(mode, loader, model, device, roc=False):
    y_true = []
    y_pred = []
    predicted_prob = []
    
    for x, y in loader:
        # added line to prevent memory overflow
        with torch.no_grad():
            if mode == 'image':
                x = x.to(device)
                y = y.to(device)
                output = model(image=x)
            elif mode == 'clinical':
                x = x.to(device)
                y = y.to(device)
                output = model(tab=x)
            elif mode == 'image_and_clinical':
                image = x['image']
                record = x['tabular_info']

                image = image.to(device)
                record = record.to(device)
                y = y.to(device)
                output = model(image=image, tab=record)
            else:
                raise NotImplementedError

        y_true.append(y.cpu())
        y_pred.append(output.argmax(1).cpu())
        predicted_prob.append(torch.softmax(output, dim=1).cpu())
        
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    predicted_prob = torch.cat(predicted_prob).detach().numpy()

    auc = roc_auc_score(y_true, predicted_prob[:, 1])
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)

    if roc:
        fpr, tpr, threshold = roc_curve(y_true, predicted_prob[:, 1])
        pre, reca, pr_threshold = precision_recall_curve(y_true, predicted_prob[:, 1])
        return auc, f1, precision, recall, specificity, tn, fp, fn, tp, fpr, tpr, threshold, pre, reca, pr_threshold
    else:
        return auc, f1, precision, recall, specificity, tn, fp, fn, tp


def get_imratio(images, nclasses):
    count = {}
    for i in range(nclasses):
        count[i] = 0

    for item in images:
        image, label = item
        count[label] += 1                                                  
    
    weight_per_class = [0.] * nclasses
    N = float(sum(count.values()))
    for i in range(nclasses):
        if count[i] != 0:
            weight_per_class[i] = N / float(count[i])
        else:
            # Handle the case when count[i] is zero
            weight_per_class[i] = 0.0  # Assign a default weight value or handle it accordingly

    imratio = weight_per_class[0] / (weight_per_class[0] + weight_per_class[1])
    return imratio

def make_weights_for_balanced_classes(images, nclasses):
    count = {}
    for i in range(nclasses):
        count[i] = 0

    for item in images:
        image, label = item
        count[label] += 1
    
    weight_per_class = [0.] * nclasses
    N = float(sum(count.values()))
    for i in range(nclasses):
        if count[i] != 0:
            weight_per_class[i] = N / float(count[i])
        else:
            # Handle the case when count[i] is zero
            weight_per_class[i] = 0.0  # Assign a default weight value or handle it accordingly   

    print('weight per class', weight_per_class)

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        image, label = val
        weight[idx] = weight_per_class[label]

    return weight

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush() # force to write everything in the buffer of standard out to the terminal

    def flush(self):
        self.stdout.flush()
        self.file.flush()