import argparse
import os
import sys
import numpy as np
import pandas as pd 
import torch

from PIL import Image 
import knee_net,util_fn, dataset

import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def scale_np_from_tensor(cam, target_size=None):
    if target_size is None:
        result = np.zeros(cam.shape)
    else:
        result = np.zeros((target_size[0], target_size[1], cam.shape[-1]))
    for i in range(cam.shape[-1]):
        img = cam[:, :, i]
        min_ = np.min(img)
        max_ = np.max(img)
        img = img - min_
        img = img / (max_ - min_)

        if target_size is not None:
            img = cv2.resize(img, target_size)
        result[:, :, i] = img
    result = np.uint8(255*result)

    return result


def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (np.max(img)-np.min(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


def visualise_CAM(index, class_gradients, feature_maps, all_x):
    current_sample_index = index
    weight = class_gradients[current_sample_index]
    weight = torch.mean(weight, dim=(1,2))
    
    feature_map = feature_maps[current_sample_index]
    cam = weight[:, None, None] * feature_map

    cam = torch.sum(cam, dim=0).unsqueeze(0)
    cam = torch.nn.functional.relu(cam)
    
    image = all_x[current_sample_index].detach().cpu().numpy()
    cam = scale_cam_image(cam.detach().numpy(), target_size=(224,224))

    image = scale_cam_image(image)
    image = np.transpose(image, (1,2,0))
    visualization = show_cam_on_image(image, cam[0, :, :], use_rgb=True)

    CAM = np.uint8(255*cam[0, :, :])
    CAM = cv2.merge([CAM, CAM, CAM])
    images = np.hstack((np.uint8(255*image), visualization))
    return images 

# for convnext
def extract_gradients_image_metadata(model, x, tab, y):
    with torch.enable_grad():
        feature_maps = model.image_net.model.features(x).detach()
        feature_maps.requires_grad_(True)
        outputs = model.image_net.model.avgpool(feature_maps)
        outputs = torch.flatten(outputs, 1)
        
        tab = model.tab_net.featurizer(tab)
        result = model.classifier(torch.cat([outputs, tab], dim=1))

        targets = [ClassifierOutputTarget(y[i].item()) for i in range(y.size(0))]
        loss = sum([target(output) for target, output in zip(targets, result)])

        loss.backward(retain_graph=True)
        gradient = feature_maps.grad

        gradient = gradient.clone()
        return feature_maps, gradient, result

def extract_gradients_image(model, x, y):
    with torch.enable_grad():
        feature_maps = model.image_net.model.features(x).detach()
        feature_maps.requires_grad_(True)
        outputs = model.image_net.model.avgpool(feature_maps)
        outputs = torch.flatten(outputs, 1)
        result = model.classifier(outputs)

        targets = [ClassifierOutputTarget(y[i].item()) for i in range(y.size(0))]
        loss = sum([target(output) for target, output in zip(targets, result)])

        loss.backward(retain_graph=True)
        gradient = feature_maps.grad

        gradient = gradient.clone()
        return feature_maps, gradient, result

def get_index_in_dataset(df, id):
    return df.index[df['IC']==id].tolist()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the GradCAM of images using neural network.')
    parser.add_argument('--mode', type=str, choices=['image', 'image_and_clinical'])
    parser.add_argument('--keys', type=list, default=util_fn.keys_of_interest)
    parser.add_argument('--label', type=str)
    parser.add_argument('--mcid', type=float)
    parser.add_argument('--output_dir', type=str, default='')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_features', type=int, default=60)
    parser.add_argument('--net', type=str, default='convnext_tiny', choices=['resnet18', 'resnet101', 'resnet152', 'alexnet', 'convnext_tiny', 'convnext_base', 'vit_b_32'])

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--id_of_interest', type=str, nargs='+') # get the list of patient ids that we want to see GradCAM

    args = parser.parse_args()

    output_dir = args.mode + '_' + args.label
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load model
    model = knee_net.KneeNet(args.mode, False, args.num_features, [1024, 512, 256, 128, 32], [0.001, 0.01, 0.01, 0.01, 0.01], 0.1, 2, args.net)

    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    # load data
    train_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.train_file))
    test_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.test_file))
    preprocessor = util_fn.encode_category_columns(util_fn.categorical_cols, util_fn.get_feature(train_df, args.keys))
    train_dataset = dataset.KneeDataset(train_df, args.mode, args.label, args.mcid, args.keys, transform=False, preprocessor=preprocessor)
    test_dataset = dataset.KneeDataset(test_df, args.mode, args.label, args.mcid, args.keys, transform=False, preprocessor=preprocessor)

    # get cam
    for id in args.id_of_interest:
        # get data

        if id in train_df['IC'].values:
            indices = get_index_in_dataset(train_df, id)
            id_data = [train_dataset[idx] for idx in indices]
            tag = 'train'
        else:
            indices = get_index_in_dataset(test_df, id)
            id_data = [test_dataset[idx] for idx in indices]
            tag = 'test'

        # get feature map and gradients 
        labels = [i[1] for i in id_data]
        labels = torch.LongTensor(labels)
        if args.mode == 'image_and_clinical':
            images = [i[0]['image'] for i in id_data]
            records = [i[0]['tabular_info'] for i in id_data]
            
            images = torch.stack(images, dim=0)
            records = torch.stack(records, dim=0)
            
            feature_maps, gradient, y_pred = extract_gradients_image_metadata(model, images, torch.Tensor(records), labels)
        else:
            images = [i[0] for i in id_data]           
            images = torch.stack(images, dim=0)

            feature_maps, gradient, y_pred = extract_gradients_image(model, images, labels)
            
        # store cam 
        for index in range(len(indices)):
            img = visualise_CAM(index, gradient, feature_maps, images)
            name = id + '_' + tag + str(indices[index]) +'.jpg'
            img = Image.fromarray(img)
            img.save(os.path.join(output_dir, name))