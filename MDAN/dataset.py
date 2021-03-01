import os
import csv
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

# Dataset
class WholeImageData(Dataset):
    
    def __init__(self,root,scale,transforms = None,Ds=[0,1,2,3]):
        #從csv讀取所有的Data
        img_paths  = []
        cls_labels = []
        csv_root = [os.path.join(root,name,(name+'_train.csv')) for name in sorted(os.listdir(root))] 
        for i in Ds:
            img1=[]
            label1=[]
            with open(csv_root[i], newline='') as csvfile:
                rows = csv.reader(csvfile)
                for row in rows:
                    if row[0]=='image_name':
                        continue
                    img1.append(row[0])
                    label1.append(row[1])

            random_order = np.arange(len(img1))
            np.random.shuffle(random_order)

            img1 = np.array(img1)
            label1 = np.array(label1)
            img1 = img1[random_order]
            label1 = label1[random_order]

            img_paths.append(img1)
            cls_labels.append(label1)

        #產生對應的路徑與資訊
        self.img_paths  = img_paths
        self.cls_label = cls_labels
        self.scale     = scale
        self.root      = root
        self.Ds=Ds
        self.domain_name=[name for name in sorted(os.listdir(root))]
        #照片做變化
        self.transforms = transforms
        
    def __getitem__(self,index):
        xs=[]
        ys=[]
        for i in range(4):
            img_path = self.img_paths[i][index]
            cls_label = int(self.cls_label[i][index])
            img = Image.open(os.path.join(self.root,img_path)).convert('RGB')
            if self.transforms:
                img = img.resize( (self.scale, self.scale), Image.BILINEAR )
                img = self.transforms(img)
#                 cls_label = np.array(cls_label,dtype = int)
            xs.append(img)
            ys.append(cls_label)
            
        return xs,ys
    
    def __len__(self):
        return min([len(self.img_paths[i]) for i in range(len(self.Ds))])

    
    def __len__(self):
        return min([len(self.img_paths[i]) for i in range(len(self.Ds))])
class TestImageData(Dataset):
    
    def __init__(self,root,scale,transforms = None,Dt=2):
        #從csv讀取所有的Data
        img_paths  = []
        cls_labels = []
        csv_root = [os.path.join(root,name,(name+'_train.csv')) for name in sorted(os.listdir(root))] 
     
        img_paths=[]
        cls_labels=[]
        with open(csv_root[Dt], newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0]=='image_name':
                    continue
                img_paths.append(row[0])
                cls_labels.append(row[1])
            


        img_paths = np.array(img_paths)
        cls_labels = np.array(cls_labels)


        #產生對應的路徑與資訊
        self.img_paths  = img_paths
        self.cls_label = cls_labels
        self.scale     = scale
        self.root      = root
        self.Dt=Dt
        self.domain_names=[name for name in sorted(os.listdir(root))]
        self.domain_name=self.domain_names[Dt]
        #照片做變化
        self.transforms = transforms
        
    def __getitem__(self,index):
        img_path = self.img_paths[index]
        cls_label = int(self.cls_label[index])
        img = Image.open(os.path.join(self.root,img_path)).convert('RGB')
        if self.transforms:
            img = img.resize( (self.scale, self.scale), Image.BILINEAR )
            img = self.transforms(img)

            
        return img,cls_label
    def __len__(self):
        return len(self.img_paths)
        
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