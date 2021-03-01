import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image

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

def get_dis_loss(dis_fake, dis_real):
    D_loss = torch.mean(dis_fake ** 2) + torch.mean((dis_real - 1) ** 2)
    return D_loss

def get_confusion_loss(dis_common):
    confusion_loss = torch.mean((dis_common - 0.5) ** 2)
    return confusion_loss

def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

class Extractor(nn.Module):
    def __init__(self , model):
        super(Extractor, self).__init__()
        #取掉model的最後1層
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.resnet_layer(x)
        x = torch.squeeze(torch.squeeze(x,dim=2),dim=2)       
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes=345):
        super(Classifier,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features= 2048, out_features = 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=4096,out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024,out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )
    def forward(self,input):
        pred_clss = self.classifier(input)
        return pred_clss

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features= 2048, out_features = 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024,out_features=2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self,input):
        pred_faketrue = self.discriminator(input)
        return pred_faketrue

def test():
    ########################################################################################
    #設定model
    extractor                  = Extractor(models.resnet101(pretrained=True)).to(device)

    source1_classifier         = Classifier(num_classes=345).to(device)
    source2_classifier         = Classifier(num_classes=345).to(device)
    source3_classifier         = Classifier(num_classes=345).to(device)

    source1_test_discriminator = Discriminator().to(device)
    source2_test_discriminator = Discriminator().to(device)
    source3_test_discriminator = Discriminator().to(device)
    ########################################################################################
    #設定dataloader
    batch_size  = 16
    num_workers = 4
    img_scale = 224
    root = './dataset_public/'
    transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    train_csv_root = [os.path.join(root,name,(name+'_train.csv')) for name in sorted(os.listdir(root))] 
    
    source1_csv_root = train_csv_root[0]
    source2_csv_root = train_csv_root[1]
    source3_csv_root = train_csv_root[3]

    valid_csv_root   = train_csv_root[2]

    s1_dataset    = ImageData(root,source1_csv_root,scale=img_scale,transforms=transform)
    s2_dataset    = ImageData(root,source2_csv_root,scale=img_scale,transforms=transform)
    s3_dataset    = ImageData(root,source3_csv_root,scale=img_scale,transforms=transform)

    valid_dataset = ImageData(root,valid_csv_root,scale=img_scale,transforms=transform)

    test_dataset  = ImageDataNoLabel(root,'test',scale=img_scale,transforms=transform)

    s1_loader = DataLoader(s1_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False) 
    s2_loader = DataLoader(s2_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False) 
    s3_loader = DataLoader(s3_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False) 
   
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False) 
   
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False) 
    

    ########################################################################################
    s1_weight = 1/3
    s2_weight = 1/3
    s3_weight = 1/3
    threshold = 0.9
    lr = 0.00001
    beta1 = 0.9
    beta2 = 0.999
    cls_epoches = 10
    gan_epoches = 5
    ########################################################################################
    #train的參數設定
    steps = 8
    #count = 0
    max_correct = 0
    max_step  = 0
    max_epoch = 0
    for step in range(steps):
        ########################################################################################
        # Part 1: assign psudo-labels to t-domain and update the label-dataset
        print ("===================== Part1 =====================")
        extractor.eval()
        source1_classifier.eval()
        source2_classifier.eval()
        source3_classifier.eval()
        
        if step > 0 :
            s1_weight = s1_weight_loss / (s1_weight_loss + s2_weight_loss + s3_weight_loss)
            s2_weight = s2_weight_loss / (s1_weight_loss + s2_weight_loss + s3_weight_loss)
            s3_weight = s3_weight_loss / (s1_weight_loss + s2_weight_loss + s3_weight_loss)
        print('s1_weight is {}'.format(s1_weight))
        print('s2_weight is {}'.format(s2_weight))
        print('s3_weight is {}'.format(s3_weight))

        total_v_pred = []
        for i , (v_img , v_cls_label , v_img_path) in enumerate(valid_loader):
            with torch.no_grad():
                v_img = v_img.to(device)
            
                v_feature = extractor(v_img)

                s1_cls = source1_classifier(v_feature)
                s2_cls = source2_classifier(v_feature)
                s3_cls = source3_classifier(v_feature)
                
                v_pred = s1_cls * s1_weight + s2_cls * s2_weight + s3_cls * s3_weight
                for small_v_pred in v_pred:
                    total_v_pred.append(small_v_pred.tolist())
        total_v_pred = torch.tensor(total_v_pred)
        if step == 0 :   
            pse_ids = total_v_pred.data.max(1, keepdim=True)[1].cpu().numpy()
        else:
            ids = total_v_pred.data.max(1, keepdim=True)[1].cpu().numpy()
            for j in range(ids.shape[0]):
                if total_v_pred[j,ids[j]] >= threshold:
                    pse_ids[j] = ids[j]
        ########################################################################################                
        # Part 2: train F1t, F2t with pseudo labels
        print ("===================== Part2 =====================")
        extractor.train()
        source1_classifier.train()
        source2_classifier.train()
        source3_classifier.train()
        
        v_pse_dataset = PseImageData(root,valid_csv_root,pse_ids.squeeze(axis=1).tolist(),scale=img_scale,transforms=transform)
        v_pse_dataloader = DataLoader(v_pse_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False) 

        optim_extract = torch.optim.Adam(extractor.parameters(),lr=lr, betas=(beta1, beta2))
        optim_s1_cls = torch.optim.Adam(source1_classifier.parameters(), lr=lr, betas=(beta1, beta2))
        optim_s2_cls = torch.optim.Adam(source2_classifier.parameters(), lr=lr, betas=(beta1, beta2))
        optim_s3_cls = torch.optim.Adam(source3_classifier.parameters(), lr=lr, betas=(beta1, beta2))

        for cls_epoch in range(cls_epoches):
            for i,(source1,source2,source3,pse) in enumerate(zip(s1_loader,s2_loader,s3_loader,v_pse_dataloader)):
                s1_img , s1_cls_label , _ = source1
                s2_img , s2_cls_label , _ = source2
                s3_img , s3_cls_label , _ = source3
                pse_img , pse_t_label , _ , pse_label = pse

                s1_img , s1_cls_label = s1_img.to(device) , s1_cls_label.to(device)
                s2_img , s2_cls_label = s2_img.to(device) , s2_cls_label.to(device)
                s3_img , s3_cls_label = s3_img.to(device) , s3_cls_label.to(device)
                pse_img , pse_label   = pse_img.to(device) , pse_label.to(device)

                s1_t_imgs = torch.cat((s1_img,pse_img),dim=0)
                s2_t_imgs = torch.cat((s2_img,pse_img),dim=0)
                s3_t_imgs = torch.cat((s3_img,pse_img),dim=0)

                s1_t_labels = torch.cat((s1_cls_label,pse_label),dim=0)
                s2_t_labels = torch.cat((s2_cls_label,pse_label),dim=0)
                s3_t_labels = torch.cat((s3_cls_label,pse_label),dim=0)

                optim_extract.zero_grad()
                optim_s1_cls.zero_grad()
                optim_s2_cls.zero_grad()
                optim_s3_cls.zero_grad()

                s1_t_features = extractor(s1_t_imgs)
                s2_t_features = extractor(s2_t_imgs)
                s3_t_features = extractor(s3_t_imgs)

                s1_t_cls = source1_classifier(s1_t_features)
                s2_t_cls = source2_classifier(s2_t_features)
                s3_t_cls = source3_classifier(s3_t_features)
                
                s1_t_cls_loss = get_cls_loss(s1_t_cls , s1_t_labels)
                s2_t_cls_loss = get_cls_loss(s2_t_cls , s2_t_labels)
                s3_t_cls_loss = get_cls_loss(s3_t_cls , s3_t_labels)

                s1_t_cls_loss.backward()
                s2_t_cls_loss.backward()
                s3_t_cls_loss.backward()
                
                optim_extract.step()
                optim_s1_cls.step()
                optim_s2_cls.step()
                optim_s3_cls.step()
                
            extractor.eval()
            source1_classifier.eval()
            source2_classifier.eval()
            source3_classifier.eval()
            n_total   = 0.0
            n_correct = 0.0
            for i , (v_img , v_cls_label , v_img_path) in enumerate(valid_loader):
                v_img = v_img.to(device)
                img_feature = extractor(v_img)
                
                s1_cls = source1_classifier(img_feature)
                s2_cls = source2_classifier(img_feature)
                s3_cls = source3_classifier(img_feature)

                v_pred = s1_cls * s1_weight + s2_cls * s2_weight + s3_cls * s3_weight
                pred = v_pred.data.max(1, keepdim=True)[1].cpu()
                n_correct += pred.eq(v_cls_label.data.view_as(pred)).cpu().sum().item()
                n_total += len(v_cls_label)
            acc = n_correct * 1.0 / n_total
            print ('Accuracy of the Valid_dataset: %f' % (acc))

            if acc >= max_correct:
                max_correct = acc
                max_step    = step
                max_epoch   = cls_epoch
                torch.save(extractor.state_dict(),os.path.join('model_para',"extractor_" + str(step) + "_" + str(cls_epoch) + ".pth"))
                torch.save(source1_classifier.state_dict(),os.path.join('model_para',"s1_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
                torch.save(source2_classifier.state_dict(),os.path.join('model_para',"s2_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
                torch.save(source3_classifier.state_dict(),os.path.join('model_para',"s3_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
        
        # Part 3: train discriminator and generate mix feature 
        print ("===================== Part3 =====================")
        extractor.train()
        source1_classifier.train()
        source2_classifier.train()
        source3_classifier.train()
        source1_test_discriminator.train()
        source2_test_discriminator.train()
        source3_test_discriminator.train()

        optim_extract = torch.optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))
        optim_s1_t_dis = torch.optim.Adam(source1_test_discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        optim_s2_t_dis = torch.optim.Adam(source2_test_discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        optim_s3_t_dis = torch.optim.Adam(source3_test_discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        s1_weight_loss = 0
        s2_weight_loss = 0
        s3_weight_loss = 0
        for gan_epoch in range(gan_epoches):
            for i,(source1,source2,source3,valid) in enumerate(zip(s1_loader,s2_loader,s3_loader,valid_loader)):
                s1_img , s1_cls_label , _ = source1
                s2_img , s2_cls_label , _ = source2
                s3_img , s3_cls_label , _ = source3
                t_img , t_cls_label , _  = valid

                s1_img , s1_cls_label = s1_img.to(device) , s1_cls_label.to(device)
                s2_img , s2_cls_label = s2_img.to(device) , s2_cls_label.to(device)
                s3_img , s3_cls_label = s3_img.to(device) , s3_cls_label.to(device)
                t_img  ,  t_cls_label   = t_img.to(device)  ,  t_cls_label.to(device)

                extractor.zero_grad()
                s1_feature = extractor(s1_img)
                s2_feature = extractor(s2_img)
                s3_feature = extractor(s3_img)
                t_feature  = extractor(t_img)
                
                s1_cls = source1_classifier(s1_feature)
                s2_cls = source1_classifier(s2_feature)
                s3_cls = source1_classifier(s3_feature)
                
                s1_t_fake = source1_test_discriminator(s1_feature)
                s1_t_real = source1_test_discriminator(t_feature)

                s2_t_fake = source2_test_discriminator(s2_feature)
                s2_t_real = source2_test_discriminator(t_feature)

                s3_t_fake = source2_test_discriminator(s3_feature)
                s3_t_real = source2_test_discriminator(t_feature)

                s1_cls_loss = get_cls_loss(s1_cls, s1_cls_label)
                s2_cls_loss = get_cls_loss(s2_cls, s2_cls_label)
                s3_cls_loss = get_cls_loss(s3_cls, s3_cls_label)

                s1_t_dis_loss = get_dis_loss(s1_t_fake, s1_t_real)
                s2_t_dis_loss = get_dis_loss(s2_t_fake, s2_t_real)
                s3_t_dis_loss = get_dis_loss(s3_t_fake, s3_t_real)

                s1_weight_loss += s1_t_dis_loss.data[0]
                s2_weight_loss += s2_t_dis_loss.data[0]
                s3_weight_loss += s3_t_dis_loss.data[0]

                if (s1_t_dis_loss.data[0] > s2_t_dis_loss.data[0]) and (s1_t_dis_loss.data[0] > s3_t_dis_loss.data[0]):
                    
                    s1_t_confusion_loss_s1 = get_confusion_loss(s1_t_fake)
                    s1_t_confusion_loss_t  = get_confusion_loss(s1_t_real)
                    s1_t_confusion_loss    = 0.5 * s1_t_confusion_loss_s1 + 0.5 * s1_t_confusion_loss_t
                    
                    s1_cls_loss.backward()
                    s2_cls_loss.backward()
                    s3_cls_loss.backward()
                    s1_t_confusion_loss.backward()

                elif (s2_t_dis_loss.data[0] > s1_t_dis_loss.data[0]) and (s2_t_dis_loss.data[0] > s3_t_dis_loss.data[0]):
                    
                    s2_t_confusion_loss_s2 = get_confusion_loss(s2_t_fake)
                    s2_t_confusion_loss_t  = get_confusion_loss(s2_t_real)
                    s2_t_confusion_loss    = 0.5 * s2_t_confusion_loss_s2 + 0.5 * s2_t_confusion_loss_t
                    
                    s1_cls_loss.backward()
                    s2_cls_loss.backward()
                    s3_cls_loss.backward()
                    s2_t_confusion_loss.backward()

                elif (s3_t_dis_loss.data[0] > s1_t_dis_loss.data[0]) and (s3_t_dis_loss.data[0] > s2_t_dis_loss.data[0]):
                    
                    s3_t_confusion_loss_s3 = get_confusion_loss(s3_t_fake)
                    s3_t_confusion_loss_t  = get_confusion_loss(s3_t_real)
                    s3_t_confusion_loss    = 0.5 * s3_t_confusion_loss_s3 + 0.5 * s3_t_confusion_loss_t

                    s1_cls_loss.backward()
                    s2_cls_loss.backward()
                    s3_cls_loss.backward()
                    s3_t_confusion_loss.backward()

                optim_extract.step()
                
                source1_test_discriminator.zero_grad()
                source2_test_discriminator.zero_grad()
                source3_test_discriminator.zero_grad()

                s1_t_fake = source1_test_discriminator(s1_feature.detach())
                s1_t_real = source1_test_discriminator(t_feature.detach())

                s2_t_fake = source2_test_discriminator(s2_feature.detach())
                s2_t_real = source2_test_discriminator(t_feature.detach())

                s3_t_fake = source3_test_discriminator(s3_feature.detach())
                s3_t_real = source3_test_discriminator(t_feature.detach())

                s1_t_dis_loss = get_dis_loss(s1_t_fake, s1_t_real)
                s2_t_dis_loss = get_dis_loss(s2_t_fake, s2_t_real)
                s3_t_dis_loss = get_dis_loss(s3_t_fake, s3_t_real)

                s1_t_dis_loss.backward()
                s2_t_dis_loss.backward()
                s3_t_dis_loss.backward()

                optim_s1_t_dis.step()
                optim_s2_t_dis.step()
                optim_s3_t_dis.step()
    print("max_correct is :",str(max_correct))
    print("max_step is :",str(max_step+1))
    print("max_epoch is :",str(max_epoch+1))
    ########################################################################################


    ########################################################################################
    #測試model
    # x = torch.randn(15,3,224,224).to(device)

    # feature = extractor(x)
    # print(feature.shape)

    # cls_label = source1_classifier(feature)
    # print(cls_label.shape)

    # fakeortrue = source1_test_discriminator(feature)
    # print(fakeortrue.shape)
    ########################################################################################



if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1214)
    np.random.seed(1214)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('===========Device used :', device,'===========')
   
    test()