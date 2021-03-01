import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#######################################

class ImageData(Dataset):
    def __init__(self,root,csv_root,scale,transforms = None):
        #從csv讀取所有的Data
        img_paths  = []
        cls_labels = []
        with open(csv_root, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0]=='image_name':
                    continue
                img_paths.append( row[0])
                cls_labels.append(row[1])
        #產生對應的路徑與資訊
        self.img_path  = [ img_path for  img_path in  img_paths] 
        self.cls_label = [cls_label for cls_label in cls_labels]
        self.scale     = scale
        self.root      = root
        #照片做變化
        self.transforms = transforms    
    def __getitem__(self,index):
        img_path = self.img_path[index]
        cls_label = self.cls_label[index]
        
        img = Image.open(os.path.join(self.root,img_path)).convert('RGB')
        if self.transforms:
            img = img.resize( (self.scale, self.scale), Image.BILINEAR )
            img = self.transforms(img)
            # check tv.transforms.ToPILImage()(img).convert('RGB')
            cls_label = np.array(cls_label,dtype = int)     
        return img , cls_label , img_path
    def __len__(self):
        return len(self.img_path)

class PseImageData(Dataset):
    def __init__(self,root,csv_root,pse_label,scale,transforms = None):
        #從csv讀取所有的Data
        img_paths  = []
        cls_labels = []
        with open(csv_root, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0]=='image_name':
                    continue
                img_paths.append( row[0])
                cls_labels.append(row[1])
        #產生對應的路徑與資訊
        self.img_path  = [ img_path for  img_path in  img_paths] 
        self.cls_label = [cls_label for cls_label in cls_labels]
        self.pse_labels = pse_label
        self.scale     = scale
        self.root      = root
        #照片做變化
        self.transforms = transforms    
    def __getitem__(self,index):
        img_path = self.img_path[index]
        cls_label = self.cls_label[index]
        pse_label = self.pse_labels[index]
        img = Image.open(os.path.join(self.root,img_path)).convert('RGB')
        if self.transforms:
            img = img.resize( (self.scale, self.scale), Image.BILINEAR )
            img = self.transforms(img)
            # check tv.transforms.ToPILImage()(img).convert('RGB')
            cls_label = np.array(cls_label,dtype = int)     
        return img , cls_label , img_path , pse_label
    def __len__(self):
        return len(self.img_path)

class ImageDataNoLabel(Dataset):
    
    def __init__(self,root,test_root,scale,transforms = None):
        #從csv讀取所有的Data
        img_names = sorted(os.listdir(os.path.join(root,test_root)))
        #產生對應的路徑與資訊
        self.img_paths  = [ os.path.join(root,test_root,img_name) for img_name in img_names] 
        self.label_names= [      os.path.join(test_root,img_name) for img_name in img_names] 
        self.scale     = scale
        #照片做變化
        self.transforms = transforms
        
    def __getitem__(self,index):
        img_path = self.img_paths[index]
        label_name = self.label_names[index]
        
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = img.resize( (self.scale, self.scale), Image.BILINEAR )
            img = self.transforms(img)
            # check tv.transforms.ToPILImage()(img).convert('RGB')
            
        return img , label_name
    
    def __len__(self):
        return len(self.img_paths)

#######################################

class Extractor(nn.Module):
    def __init__(self , model):
        super(Extractor, self).__init__()
        #取掉model的最後1層
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.resnet_layer(x)    
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes=345):
        super(Classifier,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features= 2048, out_features = 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=4096,out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(in_features=2048,out_features=num_classes),
            #nn.LogSoftmax(dim=1)
        )
    def forward(self,input):
        pred_clss = self.classifier(input.view(input.shape[0],-1))
        return pred_clss

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features= 2048, out_features = 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=1024,out_features=1),
            #nn.LogSoftmax(dim=1)
        )
    def forward(self,input):
        pred_faketrue = self.discriminator(input.view(input.shape[0],-1))
        return pred_faketrue

#######################################

def get_dis_loss(dis_fake, dis_real):
    D_loss = torch.mean(dis_fake ** 2) + torch.mean((dis_real - 1) ** 2)
    return D_loss

def get_confusion_loss(dis_common):
    confusion_loss = torch.mean((dis_common - 0.5) ** 2)
    return confusion_loss

def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred,dim=1), gt)
    return cls_loss

#######################################

def test():
    x = torch.randn(5,3,224,224)

    extractor = Extractor(models.resnet101(pretrained=True))
    # 輸出是 5 * 2048 * 1 * 1
    classifier = Classifier()
    # 輸出是 5 * 345
    discriminator = Discriminator()
    # 輸出是 5 * 1

    print(extractor(x).shape)
    print(classifier(extractor(x)).shape)
    print(discriminator(extractor(x)).shape)

if __name__ == '__main__':
    test()