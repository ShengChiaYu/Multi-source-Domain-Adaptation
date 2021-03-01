import os
import csv
import torch
import argparse
import numpy as np
from torch import nn
from PIL import Image
import torchvision as tv
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        #取掉model的最後1層
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.resnet_layer(x)
        x = torch.squeeze(torch.squeeze(x,dim=2),dim=2)       
        return x

class CNNModel(nn.Module):
    
    def __init__(self):
        super(CNNModel,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features= 2048, out_features = 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=4096,out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024,out_features=345),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,input):
        pred_clss = self.classifier(input)
        return pred_clss

class WholeImageData(Dataset):
    
    def __init__(self,root,scale,transforms = None):
        #從csv讀取所有的Data
        img_paths  = []
        cls_labels = []
        csv_root = [os.path.join(root,name,(name+'_train.csv')) for name in sorted(os.listdir(root))] 
        for i in [0,1,3]:
            with open(csv_root[i], newline='') as csvfile:
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

def test():
    resnet = models.resnet101(pretrained=True)
    resnet = Net(resnet).eval().to(device)

    net = CNNModel().to(device)

    num_workers = 0
    batch_size  = 5

    root = './dataset_public/'
    train_csv_root = [os.path.join(root,name,(name+'_train.csv')) for name in sorted(os.listdir(root))] 

    transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    dataset    = ImageData(root,train_csv_root[0],scale=224,transforms=transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False) 

    dataiter = iter(dataloader)
    img , cls_label , img_path = dataiter.next()

    print(img.shape , cls_label , img_path)
    # plt.figure()
    # plt.imshow(tv.transforms.ToPILImage()(img[0]).convert('RGB'))
    # plt.savefig("test.png")
    
    criterion = torch.nn.CrossEntropyLoss()

    img       = img.to(device)
    cls_label = cls_label.long().to(device)

    feature = resnet(img)
    output  = net(feature) 
    loss = criterion(output,cls_label)

    print(feature.shape,output.shape,loss)

def train(pre_model,model,train_data_loader,valid_data_loader,opts):

    if not os.path.exists(opts.model_dir):
        os.makedirs(opts.model_dir, exist_ok=True)

    if opts.vis:
        import visdom
        vis = visdom.Visdom(env='Train_Final_Model')
    
    optimizer = torch.optim.Adam(model.parameters(),lr=opts.lr,weight_decay=opts.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    for ep in range(opts.epoch):
        print('==========================================')
        model.train()
        total_loss = 0.0
        for i,(img,cls_label,_) in enumerate(train_data_loader):
            
            img , cls_label = img.to(device) , cls_label.long().to(device)

            with torch.no_grad():
                feature = pre_model(img).detach()

            optimizer.zero_grad() 
            output  = model(feature)

            loss = criterion(output,cls_label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print('Train Epoch : {}\t Average_Loss: {:.6f}'.format(ep,total_loss/len(train_data_loader)))
        model.eval()
        val_loss   = 0.0
        n_correct = 0.0
        n_total = 0.0
        for i,(v_img,v_cls_label,_) in enumerate(valid_data_loader):
            with torch.no_grad():
                v_img , v_cls_label = v_img.to(device) , v_cls_label.long().to(device)
                v_feature = pre_model(v_img).detach()
                
                v_output  = model(v_feature)
                
                pred = v_output.data.max(1, keepdim=True)[1]
                n_correct += pred.eq(v_cls_label.data.view_as(pred)).cpu().sum().item()
                n_total += len(v_cls_label)

                loss = criterion(v_output,v_cls_label)
                val_loss += loss.item()

        acc = n_correct * 1.0 / n_total
        print('Valid  Epoch : {}\t Average_Loss:  {:.6f}'.format(ep,val_loss/len(valid_data_loader)))
        print ('Accuracy of the Valid_dataset: %f' % (acc))
        if opts.vis:
            if ep==0:
                vis.line(X=[ep],Y=[total_loss/len(train_data_loader)],win='Final Project',opts={'title':'Final Project'},name='train')
                vis.line(X=[ep],Y=[val_loss/len(valid_data_loader)]   ,win='Final Project',opts={'title':'Final Project'},name='valid',update='append')
            else:
                vis.line(X=[ep],Y=[total_loss/len(train_data_loader)],win='Final Project',opts={'title':'Final Project'},name='train',update='append')
                vis.line(X=[ep],Y=[val_loss/len(valid_data_loader)]   ,win='Final Project',opts={'title':'Final Project'},name='valid',update='append')
        if (ep+1) % opts.save_interval == 0:
            print('=========save the model at Epoch : ' + str(ep) + ' =========')
            torch.save(model.state_dict(), os.path.join(opts.model_dir,'Final-epoch-{}.pth'.format(ep)))
    print('=========save the model at last Epoch   =========')
    torch.save(model.state_dict(), os.path.join(opts.model_dir,'Final-last.pth'))

def infer(pre_model,model,test_data_loader,opts):

    if not os.path.exists(opts.model_dir):
        os.makedirs(opts.model_dir, exist_ok=True)

    model.eval()
    n_pred = []
    n_name = []
    for i,(img,img_name) in enumerate(test_data_loader):
        with torch.no_grad():
            img = img.to(device)
            feature = pre_model(img).detach()
            output  = model(feature)

            pred = output.data.max(1, keepdim=True)[1]  

            for v_pred in pred:
                n_pred.append(v_pred.item())

            for v_name in img_name:
                n_name.append(v_name)

    output_path = os.path.join(opts.model_dir, opts.pred_dir)
    print('=====Write output to %s =====' % output_path)
    
    with open(output_path, 'w') as f:
        f.write('image_name,label\n')
        for i, (s_pred,s_name) in enumerate(zip(n_pred,n_name)): 
            f.write('{},{}\n'.format(s_name,s_pred))
        f.close()

def main(opts):

    resnet = models.resnet101(pretrained=True)
    resnet = Net(resnet).eval().to(device)
    net = CNNModel().to(device)
    
    map_location = lambda storage, loc: storage
    if opts.model_name:
        print('===========loading model :', opts.model_name,'===========')
        net.load_state_dict(torch.load(opts.model_name, map_location=map_location))
    
    transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    # 0 1 3 可以用  2 是 validation 4會GG 
    csv_root = [os.path.join(opts.root,name,(name+'_train.csv')) for name in sorted(os.listdir(opts.root))] 

    if opts.train:
        print('============ Start validation dataset  ============')
        valid_dataset     = ImageData(opts.root,csv_root[2],scale=224,transforms=transform)
        valid_data_loader = DataLoader(valid_dataset,batch_size=opts.batch_size,shuffle=False,
                                        num_workers=opts.num_workers,drop_last=False) 
        print('============ Start Training dataset  ============')
        train_dataset     = WholeImageData(opts.root,scale=224,transforms=transform)
        train_data_loader = DataLoader(train_dataset,batch_size=opts.batch_size,shuffle=True,
                                    num_workers=opts.num_workers,drop_last=False) 
        print('============ Strat Training ============')
        train(resnet,net,train_data_loader,valid_data_loader,opts)

    elif opts.infer:    
        print('============ Start Testing dataset  ============')
        test_dataset     = ImageDataNoLabel(opts.root,opts.test_root,scale=224,transforms=transform)
        test_data_loader = DataLoader(test_dataset,batch_size=opts.batch_size,shuffle=False,
                                    num_workers=opts.num_workers,drop_last=False) 
        print('============ Strat Infering ============') 
        infer(resnet,net,test_data_loader,opts)
    else:
        print('infer or train plz')  

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1214)
    np.random.seed(1214)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('===========Device used :', device,'===========')

    parser = argparse.ArgumentParser(description= 'DLCV-Final')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train',action='store_true',default=False,
                        dest='train',help='Input --train to Train')
    group.add_argument('--infer',action='store_true',default=False,
                        dest='infer',help='Input --infer to infer')        

    parser.add_argument('--root',type = str,
                        default = './dataset_public/',dest = 'root', help = 'Path to root')
    parser.add_argument('--test_root',type = str,
                        default = 'test',dest = 'test_root', help = 'Path to test_root')                 

    parser.add_argument('--model_dir',type=str,
                        default='./model_para/',dest= 'model_dir',
                        help= 'Path to save the model parameters')
    parser.add_argument('--pred_dir',type=str,
                        default='pred.csv',dest= 'pred_dir',
                        help= 'Path to save the pred_dir')
 
    parser.add_argument('--model_name',type=str,default=None,help= 'model_name') 
    parser.add_argument( '--vis',type=bool,default=True,help= 'visdom')   

    parser.add_argument("--epoch", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--save_interval", type=int, default=1, help="number of save interval")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")

    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate") 
    parser.add_argument("--weight_decay", type=int, default=5e-5, help="size of the weight_decay")

    opts = parser.parse_args()
    main(opts)  

    #test()