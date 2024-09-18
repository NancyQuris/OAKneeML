import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ImageNet(nn.Module):
    def __init__(self, pretrain, net='resnet18'):
        super(ImageNet, self).__init__()
        if pretrain:
            if net == 'vit_h_14':
                weights = 'IMAGENET1K_SWAG_LINEAR_V1'
            else:
                weights = 'DEFAULT'
        else:
            weights = None
        
        if net == 'resnet18':
            self.model = torchvision.models.resnet18(weights=weights)
            del self.model.fc
            self.model.fc = Identity()
            self.num_output = 512

        elif net == 'resnet101':
            self.model = torchvision.models.resnet101(weights=weights)
            del self.model.fc
            self.model.fc = Identity()
            self.num_output = 2048

        elif net == 'resnet152':
            self.model = torchvision.models.resnet152(weights=weights)
            del self.model.fc
            self.model.fc = Identity()
            self.num_output = 2048
        
        elif net == 'resnext101_64x4d':
            self.model = torchvision.models.resnext101_64x4d(weights=weights)
            del self.model.fc
            self.model.fc = Identity()
            self.num_output = 2048

        elif net == 'alexnet':
            self.model = torchvision.models.alexnet(weights=weights)
            del self.model.classifier
            self.model.classifier = Identity()
            self.num_output = 9216
        
        elif net == 'convnext_tiny':
            self.model = torchvision.models.convnext_tiny(weights=weights)
            del self.model.classifier
            self.model.classifier = Identity()
            self.num_output = 768
        
        elif net == 'convnext_base':
            self.model = torchvision.models.convnext_base(weights=weights)
            del self.model.classifier
            self.model.classifier = Identity()
            self.num_output = 1024

        elif net == 'vit_b_32':
            self.model = torchvision.models.vit_b_32(weights=weights)
            del self.model.heads
            self.model.heads = Identity()
            self.num_output = 768

        elif net == 'vit_h_14':
            self.model = torchvision.models.vit_h_14(weights=weights)
            del self.model.heads
            self.model.heads = Identity()
            self.num_output = 1280

        elif net == 'swin_v2_t':
            self.model = torchvision.models.swin_v2_t(weights=weights)
            del self.model.head
            self.model.head = Identity()
            self.num_output = 96 * 2 ** 3
        
        elif net == 'efficientnet_v2_s':
            self.model = torchvision.models.efficientnet_v2_s(weights=weights)
            del self.model.classifier
            self.model.classifier = Identity()
            self.num_output = 1280
    
    def forward(self, x):
        return self.model(x)


class TabularNet(nn.Module):
    def __init__(self, input_size, output_size, layers, drop_out_probability, use_batch_norm=True):
        super(TabularNet, self).__init__()
        self.num_output = layers[-1]
        
        self.n_cont = input_size
        sizes = [input_size] + layers + [output_size]
        activations = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None] 
        
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+drop_out_probability,activations)):
            layers += TabularNet.bn_drop_lin(n_in, n_out, bn=use_batch_norm and i!=0, p=dp, actn=act)
        
        self.featurizer = nn.Sequential(*layers[:-1])
        self.classifier = layers[-1]
    
    def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn=None):
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        
        if p != 0: layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None: layers.append(actn)
        
        return layers

    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)
        return x 


class KneeNet(nn.Module):
    def __init__(self, mode, image_pretrain, tab_input_size, tab_layers, tab_dropout, concat_dropout, output_size, net='resnet18'):
        super(KneeNet, self).__init__()
        self.net = net
        self.mode = mode 
        
        # create the featurizer 
        if self.mode == 'image' or self.mode == 'image_and_clinical':
            self.image_net = ImageNet(image_pretrain, net=net)
        if self.mode == 'clinical' or self.mode == 'image_and_clinical':
            self.tab_net = TabularNet(tab_input_size, output_size, tab_layers, tab_dropout)

        # create the classifier 
        if self.mode == 'image':
            self.classifier = nn.Linear(self.image_net.num_output, output_size)
        if self.mode == 'image_and_clinical':
            num_output = self.image_net.num_output + self.tab_net.num_output
            self.classifier = nn.Linear(num_output, output_size)
        if self.mode == 'clinical':
            self.classifier = self.tab_net.classifier
    
        self.dropout = nn.Dropout(concat_dropout)

    def forward(self, image=None, tab=None):
        if self.mode == 'image' or self.mode == 'image_and_clinical':
            feature = self.image_net(image)
            if self.net == 'convnext_tiny' or self.net == 'convnext_base':
                feature = feature.view(-1, self.image_net.num_output)
            if self.mode == 'image_and_clinical':
                tab = self.tab_net.featurizer(tab)
                feature = torch.cat([feature, tab], dim=1)
        else:
            tab = tab.float()
            feature = self.tab_net.featurizer(tab)

        return self.dropout(self.classifier(feature))