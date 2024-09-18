import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch 
import torchvision.transforms as T
from torch.utils.data import Dataset

from transform_fn import TransformFunction
import util_fn

def get_xgb_multimodal_dataset(mode, label=None, mcid=None, keys=None, train=True, val_fold=None, preprocessor=None):
        if train:
            assert val_fold is not None, 'validation fold need to be specified'
            df = pd.read_csv(util_fn.train_file)
            train_df = df.loc[df['fold']!=val_fold]
            val_df = df.loc[df['fold']==val_fold]

            train_dataset = KneeDataset(train_df, mode, label, mcid, keys, transform=False, preprocessor=preprocessor)
            val_dataset = KneeDataset(val_df, mode, label, mcid, keys, transform=False, preprocessor=preprocessor)
            return train_dataset, val_dataset
        else:
            df = pd.read_csv(util_fn.test_file)
            return KneeDataset(df, mode, label, mcid, keys, transform=False, preprocessor=preprocessor)


def get_dataset(mode, label=None, mcid=None, keys=None, train=True, val_fold=None, preprocessor=None):
        if train:
            assert val_fold is not None, 'validation fold need to be specified'
            df = pd.read_csv(util_fn.train_file)
            train_df = df.loc[df['fold']!=val_fold]
            val_df = df.loc[df['fold']==val_fold]

            train_dataset = KneeDataset(train_df, mode, label, mcid, keys, transform=True, preprocessor=preprocessor)
            val_dataset = KneeDataset(val_df, mode, label, mcid, keys, transform=False, preprocessor=preprocessor)
            return train_dataset, val_dataset
        else:
            df = pd.read_csv(util_fn.test_file)
            return KneeDataset(df, mode, label, mcid, keys, transform=False, preprocessor=preprocessor)


class KneeDataset(Dataset):
    def __init__(self, df, mode, label, mcid, keys=None, transform=True, preprocessor=None):
        assert mode == 'image' or mode == 'image_and_clinical' or mode == 'clinical'
        self.mode = mode
        self.label = label
        self.mcid = mcid
        self.meta_data = util_fn.preprocess_dataframe(df)
        self.transform = transform

        self.tabular_data = self.meta_data[keys]
        self.tabular_data = preprocessor.transform(self.tabular_data)

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, index):
        image_path = self.meta_data['image_path'].values[index]
        image = np.load(image_path)
        label = self.meta_data[self.label].values[index]
        label = 1 if label < self.mcid else 0
        
        if self.mode == 'image':
            return self.image_transform_v1(image), label
        elif self.mode == 'image_and_clinical':
            tabular_info = self.tabular_data[index]
            return {'image': self.image_transform_v1(image), 'tabular_info': torch.Tensor(tabular_info)}, label
        else:
            tabular_info = self.tabular_data[index]
            return tabular_info, label 

    def image_transform_v1(self, image):
        clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8,8)) 
        image = clahe.apply(image)
        image = Image.fromarray(np.stack([image, image, image], axis=-1))
    
        if self.transform:
            transform = T.Compose([
                T.Resize((224,224)),
                T.RandomAffine(degrees=(-10, 10), scale=(0.6, 1.2), shear=(-5, 5)),
                T.RandomAffine(degrees=(-10, 10), scale=(0.6, 1.2), shear=(-5, 5)),
                TransformFunction(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            transform =  T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        return transform(image)