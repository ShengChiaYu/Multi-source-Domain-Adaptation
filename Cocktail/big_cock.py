import os
import csv
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from cockNetwork import Extractor , Classifier , Discriminator
from cockNetwork import ImageData, PseImageData, ImageDataNoLabel
from cockNetwork import get_cls_loss, get_dis_loss, get_confusion_loss

def load_models(extractor,s1_classifier,s2_classifier,s3_classifier,opts):
    map_location = lambda storage, loc: storage
    print('===========loading model :', opts.extractor_name,'===========')
    extractor.load_state_dict(torch.load(os.path.join(opts.model_dir,opts.extractor_name), map_location=map_location))
    print('===========loading model :', opts.s1_cls_name,'===========')
    s1_classifier.load_state_dict(torch.load(os.path.join(opts.model_dir,opts.s1_cls_name), map_location=map_location))
    print('===========loading model :', opts.s2_cls_name,'===========')
    s2_classifier.load_state_dict(torch.load(os.path.join(opts.model_dir,opts.s2_cls_name), map_location=map_location))
    print('===========loading model :', opts.s3_cls_name,'===========')
    s3_classifier.load_state_dict(torch.load(os.path.join(opts.model_dir,opts.s3_cls_name), map_location=map_location))

def get_train_valid_loaders(opts):

    transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    train_csv_root = [os.path.join(opts.root,name,(name+'_train.csv')) for name in sorted(os.listdir(opts.root))] 

    source1_csv_root = train_csv_root[0]
    source2_csv_root = train_csv_root[1]
    source3_csv_root = train_csv_root[3]

    valid_csv_root   = train_csv_root[2]

    s1_dataset    = ImageData(opts.root,source1_csv_root,scale=opts.img_scale,transforms=transform)
    s2_dataset    = ImageData(opts.root,source2_csv_root,scale=opts.img_scale,transforms=transform)
    s3_dataset    = ImageData(opts.root,source3_csv_root,scale=opts.img_scale,transforms=transform)

    valid_dataset = ImageData(opts.root ,valid_csv_root ,scale=opts.img_scale,transforms=transform)

    s1_loader = DataLoader(s1_dataset,batch_size=opts.batch_size,shuffle=True,num_workers=opts.num_workers,drop_last=False) 
    s2_loader = DataLoader(s2_dataset,batch_size=opts.batch_size,shuffle=True,num_workers=opts.num_workers,drop_last=False) 
    s3_loader = DataLoader(s3_dataset,batch_size=opts.batch_size,shuffle=True,num_workers=opts.num_workers,drop_last=False) 
   
    valid_loader = DataLoader(valid_dataset,batch_size=opts.batch_size,shuffle=True,num_workers=opts.num_workers,drop_last=False) 

    return s1_loader , s2_loader , s3_loader , valid_loader

def train(extractor,s1_classifier,s2_classifier,s3_classifier,s1_discriminator,s2_discriminator,
        s3_discriminator,s1_loader,s2_loader,s3_loader,valid_loader,opts):

    if not os.path.exists(opts.model_dir):
        os.makedirs(opts.model_dir, exist_ok=True)
    
    if opts.vis:
        import visdom
        vis = visdom.Visdom(env='Train_DCTN_Model')

    transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    train_csv_root = [os.path.join(opts.root,name,(name+'_train.csv')) for name in sorted(os.listdir(opts.root))] 
    
    max_correct = 0
    max_step = 0
    max_epoch = 0
    for step in range(opts.steps):
        ########################################################################################
        # Part 1: assign psudo-labels to t-domain and update the label-dataset
        print ("===================== Part1 =====================")
        extractor.eval()
        s1_classifier.eval()
        s2_classifier.eval()
        s3_classifier.eval()

        if step == 0:
            s1_weight = opts.s1_weight
            s2_weight = opts.s2_weight
            s3_weight = opts.s3_weight
        else:
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

                s1_cls = s1_classifier(v_feature)
                s2_cls = s2_classifier(v_feature)
                s3_cls = s3_classifier(v_feature)

                s1_cls = F.softmax(s1_cls,dim=1)
                s2_cls = F.softmax(s2_cls,dim=1)
                s3_cls = F.softmax(s3_cls,dim=1)

                v_pred = s1_cls * s1_weight + s2_cls * s2_weight + s3_cls * s3_weight
                for small_v_pred in v_pred:
                    total_v_pred.append(small_v_pred.tolist())
        total_v_pred = torch.tensor(total_v_pred)
        if step == 0 :   
            pse_ids = total_v_pred.data.max(1, keepdim=True)[1].cpu().numpy()
        else:
            ids = total_v_pred.data.max(1, keepdim=True)[1].cpu().numpy()
            for j in range(ids.shape[0]):
                if total_v_pred[j,ids[j]] >= opts.threshold:
                    pse_ids[j] = ids[j]

        ########################################################################################                
        # Part 2: train F1t, F2t with pseudo labels
        print ("===================== Part2 =====================")
        extractor.train()
        s1_classifier.train()
        s2_classifier.train()
        s3_classifier.train()

        v_pse_dataset = PseImageData(opts.root,train_csv_root[2],pse_ids.squeeze(axis=1).tolist(),scale=opts.img_scale,transforms=transform)
        v_pse_dataloader = DataLoader(v_pse_dataset,batch_size=opts.batch_size,shuffle=True,num_workers=opts.num_workers,drop_last=False) 

        optim_extract = torch.optim.Adam(    extractor.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
        optim_s1_cls  = torch.optim.Adam(s1_classifier.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
        optim_s2_cls  = torch.optim.Adam(s2_classifier.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
        optim_s3_cls  = torch.optim.Adam(s3_classifier.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

        for cls_epoch in range(opts.cls_epoches):
            s1_total_loss = 0.0
            s2_total_loss = 0.0
            s3_total_loss = 0.0
            iter_s1_loader,iter_s2_loader,iter_s3_loader,iter_v_pse_dataloader = iter(s1_loader) , iter(s2_loader) , iter(s3_loader) , iter(v_pse_dataloader)
            for i,(pse_img , pse_t_label , _ , pse_label) in enumerate(iter_v_pse_dataloader):
                #每個dataloader數目其實都不一樣所以要以 valid的為主，這樣才能trian
                try:
                    s1_img, s1_cls_label , _ = iter_s1_loader.next()
                except StopIteration:
                    iter_s1_loader = iter(s1_loader)
                    s1_img, s1_cls_label , _ = iter_s1_loader.next()
                try:
                    s2_img, s2_cls_label , _ = iter_s2_loader.next()
                except StopIteration:
                    iter_s2_loader = iter(s2_loader)
                    s2_img, s2_cls_label , _ = iter_s2_loader.next()
                try:
                    s3_img, s3_cls_label , _ = iter_s3_loader.next()
                except StopIteration:
                    iter_s3_loader = iter(s3_loader)
                    s3_img, s3_cls_label , _ = iter_s3_loader.next()

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

                s1_t_cls = s1_classifier(s1_t_features)
                s2_t_cls = s2_classifier(s2_t_features)
                s3_t_cls = s3_classifier(s3_t_features)
                
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

                s1_total_loss += s1_t_cls_loss.item()
                s2_total_loss += s2_t_cls_loss.item()
                s3_total_loss += s3_t_cls_loss.item()
            print('Train Epoch : {}\t s1_Average_Loss: {:.6f}'.format(cls_epoch,s1_total_loss/len(iter_v_pse_dataloader)))
            print('s2_Average_Loss: {:.6f} s3_Average_Loss: {:.6f}'.format(s2_total_loss/len(iter_v_pse_dataloader),s3_total_loss/len(iter_v_pse_dataloader)))
            
            if opts.vis:
                if cls_epoch==0 and step ==0:
                    vis.line(X=[cls_epoch + opts.cls_epoches*step],Y=[s1_total_loss/len(iter_v_pse_dataloader)],win='Final Project',opts={'title':'Final Project'},name='s1')
                    vis.line(X=[cls_epoch + opts.cls_epoches*step],Y=[s2_total_loss/len(iter_v_pse_dataloader)],win='Final Project',opts={'title':'Final Project'},name='s2')
                    vis.line(X=[cls_epoch + opts.cls_epoches*step],Y=[s3_total_loss/len(iter_v_pse_dataloader)],win='Final Project',opts={'title':'Final Project'},name='s3')
                else:
                    vis.line(X=[cls_epoch + opts.cls_epoches*step],Y=[s1_total_loss/len(iter_v_pse_dataloader)],win='Final Project',opts={'title':'Final Project'},name='s1',update='append')
                    vis.line(X=[cls_epoch + opts.cls_epoches*step],Y=[s2_total_loss/len(iter_v_pse_dataloader)],win='Final Project',opts={'title':'Final Project'},name='s2',update='append')
                    vis.line(X=[cls_epoch + opts.cls_epoches*step],Y=[s3_total_loss/len(iter_v_pse_dataloader)],win='Final Project',opts={'title':'Final Project'},name='s3',update='append')

            extractor.eval()
            s1_classifier.eval()
            s2_classifier.eval()
            s3_classifier.eval()
            n_total   = 0.0
            n_correct = 0.0
            for i , (v_img , v_cls_label , v_img_path) in enumerate(valid_loader):
                v_img = v_img.to(device)
                img_feature = extractor(v_img)
                
                s1_cls = s1_classifier(img_feature)
                s2_cls = s2_classifier(img_feature)
                s3_cls = s3_classifier(img_feature)

                s1_cls = F.softmax(s1_cls,dim=1)
                s2_cls = F.softmax(s2_cls,dim=1)
                s3_cls = F.softmax(s3_cls,dim=1)

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
                torch.save(extractor.state_dict(),os.path.join(opts.model_dir,"extractor_" + str(step) + "_" + str(cls_epoch) + ".pth"))
                torch.save(s1_classifier.state_dict(),os.path.join(opts.model_dir,"s1_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
                torch.save(s2_classifier.state_dict(),os.path.join(opts.model_dir,"s2_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))
                torch.save(s3_classifier.state_dict(),os.path.join(opts.model_dir,"s3_cls_" + str(step) + "_" + str(cls_epoch) + ".pth"))

        # Part 3: train discriminator and generate mix feature 
        print ("===================== Part3 =====================")
        extractor.train()
        s1_classifier.train()
        s2_classifier.train()
        s3_classifier.train()
        s1_discriminator.train()
        s2_discriminator.train()
        s3_discriminator.train()

        optim_extract = torch.optim.Adam(extractor.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
        optim_s1_t_dis = torch.optim.Adam(s1_discriminator.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
        optim_s2_t_dis = torch.optim.Adam(s2_discriminator.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
        optim_s3_t_dis = torch.optim.Adam(s3_discriminator.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

        s1_weight_loss = 0
        s2_weight_loss = 0
        s3_weight_loss = 0
        for gan_epoch in range(opts.gan_epoches):
            iter_s1_loader,iter_s2_loader,iter_s3_loader,iter_valid_loader = iter(s1_loader) , iter(s2_loader) , iter(s3_loader) , iter(valid_loader)
            for i,(t_img , t_cls_label , _ ) in enumerate(iter_valid_loader):
                try:
                    s1_img , s1_cls_label , _ = iter_s1_loader.next()
                except StopIteration:
                    iter_s1_loader = iter(s1_loader)
                    s1_img , s1_cls_label , _ = iter_s1_loader.next()
                try:
                    s2_img , s2_cls_label , _ = iter_s2_loader.next()
                except StopIteration:
                    iter_s2_loader = iter(s2_loader)
                    s2_img , s2_cls_label , _ = iter_s2_loader.next()
                try:
                    s3_img , s3_cls_label , _ = iter_s3_loader.next()
                except StopIteration:
                    iter_s3_loader = iter(s3_loader)
                    s3_img , s3_cls_label , _ = iter_s3_loader.next()

                s1_img , s1_cls_label = s1_img.to(device) , s1_cls_label.to(device)
                s2_img , s2_cls_label = s2_img.to(device) , s2_cls_label.to(device)
                s3_img , s3_cls_label = s3_img.to(device) , s3_cls_label.to(device)
                t_img  ,  t_cls_label =  t_img.to(device) ,  t_cls_label.to(device)

                extractor.zero_grad()
                s1_feature = extractor(s1_img)
                s2_feature = extractor(s2_img)
                s3_feature = extractor(s3_img)
                t_feature  = extractor(t_img)
                
                s1_cls = s1_classifier(s1_feature)
                s2_cls = s2_classifier(s2_feature)
                s3_cls = s3_classifier(s3_feature)
                
                s1_t_fake = s1_discriminator(s1_feature)
                s1_t_real = s1_discriminator(t_feature)

                s2_t_fake = s2_discriminator(s2_feature)
                s2_t_real = s2_discriminator(t_feature)

                s3_t_fake = s3_discriminator(s3_feature)
                s3_t_real = s3_discriminator(t_feature)

                s1_cls_loss = get_cls_loss(s1_cls, s1_cls_label)
                s2_cls_loss = get_cls_loss(s2_cls, s2_cls_label)
                s3_cls_loss = get_cls_loss(s3_cls, s3_cls_label)

                s1_t_dis_loss = get_dis_loss(s1_t_fake, s1_t_real)
                s2_t_dis_loss = get_dis_loss(s2_t_fake, s2_t_real)
                s3_t_dis_loss = get_dis_loss(s3_t_fake, s3_t_real)

                s1_weight_loss += s1_t_dis_loss.item()
                s2_weight_loss += s2_t_dis_loss.item()
                s3_weight_loss += s3_t_dis_loss.item()

                if (s1_t_dis_loss.item() > s2_t_dis_loss.item()) and (s1_t_dis_loss.item() > s3_t_dis_loss.item()):
                    
                    s1_t_confusion_loss_s1 = get_confusion_loss(s1_t_fake)
                    s1_t_confusion_loss_t  = get_confusion_loss(s1_t_real)
                    s1_t_confusion_loss    = 0.5 * s1_t_confusion_loss_s1 + 0.5 * s1_t_confusion_loss_t
                    
                    torch.autograd.backward([s1_cls_loss, s2_cls_loss,s3_cls_loss, s1_t_confusion_loss])

                elif (s2_t_dis_loss.item() > s1_t_dis_loss.item()) and (s2_t_dis_loss.item() > s3_t_dis_loss.item()):
                    
                    s2_t_confusion_loss_s2 = get_confusion_loss(s2_t_fake)
                    s2_t_confusion_loss_t  = get_confusion_loss(s2_t_real)
                    s2_t_confusion_loss    = 0.5 * s2_t_confusion_loss_s2 + 0.5 * s2_t_confusion_loss_t
                    
                    torch.autograd.backward([s1_cls_loss, s2_cls_loss,s3_cls_loss, s2_t_confusion_loss])

                elif (s3_t_dis_loss.item() > s1_t_dis_loss.item()) and (s3_t_dis_loss.item() > s2_t_dis_loss.item()):
                    
                    s3_t_confusion_loss_s3 = get_confusion_loss(s3_t_fake)
                    s3_t_confusion_loss_t  = get_confusion_loss(s3_t_real)
                    s3_t_confusion_loss    = 0.5 * s3_t_confusion_loss_s3 + 0.5 * s3_t_confusion_loss_t

                    torch.autograd.backward([s1_cls_loss, s2_cls_loss,s3_cls_loss, s3_t_confusion_loss])

                optim_extract.step()
                
                s1_discriminator.zero_grad()
                s2_discriminator.zero_grad()
                s3_discriminator.zero_grad()

                s1_t_fake = s1_discriminator(s1_feature.detach())
                s1_t_real = s1_discriminator(t_feature.detach())

                s2_t_fake = s2_discriminator(s2_feature.detach())
                s2_t_real = s2_discriminator(t_feature.detach())

                s3_t_fake = s3_discriminator(s3_feature.detach())
                s3_t_real = s3_discriminator(t_feature.detach())

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

def predict(extractor,s1_classifier,s2_classifier,s3_classifier,opts):

    transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    test_dataset  = ImageDataNoLabel(opts.root,'test',scale=opts.img_scale,transforms=transform)
    test_loader = DataLoader(test_dataset,batch_size=opts.batch_size,shuffle=False,num_workers=opts.num_workers,drop_last=False) 

    extractor.eval()
    s1_classifier.eval()
    s2_classifier.eval()
    s3_classifier.eval()

    s1_weight = opts.s1_weight  #要自己填
    s2_weight = opts.s2_weight  #要自己填
    s3_weight = opts.s3_weight  #要自己填

    n_pred = []
    n_name = []
    for i , (img , label_name) in enumerate(test_loader):
        with torch.no_grad():
            img = img.to(device)     
            feature = extractor(img)

            s1_cls = s1_classifier(feature)
            s2_cls = s2_classifier(feature)
            s3_cls = s3_classifier(feature)

            s1_cls = F.softmax(s1_cls,dim=1)
            s2_cls = F.softmax(s2_cls,dim=1)
            s3_cls = F.softmax(s3_cls,dim=1)

            v_pred = s1_cls * s1_weight + s2_cls * s2_weight + s3_cls * s3_weight

            for small_v_pred in v_pred:
                n_pred.append(small_v_pred.tolist())
            
            for v_name in label_name:
                n_name.append(v_name)

    n_pred = torch.tensor(n_pred)
    ids = n_pred.data.max(1, keepdim=True)[1].cpu().numpy()

    output_path = os.path.join(opts.model_dir, opts.pred_dir)
    print('=====Write output to %s =====' % output_path)

    with open(output_path, 'w') as f:
        f.write('image_name,label\n')
        for i, (s_pred,s_name) in enumerate(zip(ids,n_name)): 
            f.write('{},{}\n'.format(s_name,s_pred))
        f.close()

def main(opts):
    ########################################################################################
    #設定model
    extractor        = Extractor(models.resnet101(pretrained=True)).to(device)

    s1_classifier    = Classifier(num_classes=345).to(device)
    s2_classifier    = Classifier(num_classes=345).to(device)
    s3_classifier    = Classifier(num_classes=345).to(device)

    s1_discriminator = Discriminator().to(device)
    s2_discriminator = Discriminator().to(device)
    s3_discriminator = Discriminator().to(device)
    ########################################################################################
    if opts.extractor_name:
       load_models(extractor,s1_classifier,s2_classifier,s3_classifier,opts)

    if opts.train:
        print('============ Taking train dataloaders ============')
        s1_loader , s2_loader , s3_loader , valid_loader = get_train_valid_loaders(opts)
        print('============ Start Training ============')
        train(extractor,s1_classifier,s2_classifier,s3_classifier,s1_discriminator,s2_discriminator,s3_discriminator,
                        s1_loader   , s2_loader   , s3_loader , valid_loader ,opts)
    elif opts.infer:  
        print('============ Strat predicting ============') 
        predict(extractor,s1_classifier,s2_classifier,s3_classifier,opts)
    else:
        print('infer or train plz')  

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1214)
    np.random.seed(1214)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('===========Device used :', device,'===========')

    parser = argparse.ArgumentParser(description= 'DLCV-DCTN')
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
 
    parser.add_argument('--extractor_name',type=str,default=None,help= 'extractor_name') 
    parser.add_argument('--s1_cls_name',type=str,default=None,help= 's1_cls_name') 
    parser.add_argument('--s2_cls_name',type=str,default=None,help= 's2_cls_name') 
    parser.add_argument('--s3_cls_name',type=str,default=None,help= 's3_cls_name') 

    parser.add_argument( '--vis',type=bool,default=True,help= 'visdom')   

    parser.add_argument("--epoch", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--save_interval", type=int, default=1, help="number of save interval")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")

    parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate") 
    parser.add_argument("--weight_decay", type=int, default=5e-5, help="size of the weight_decay")

    parser.add_argument("--img_scale", type=int, default=224)

    parser.add_argument("--s1_weight", default=1/3)
    parser.add_argument("--s2_weight", default=1/3)
    parser.add_argument("--s3_weight", default=1/3)

    parser.add_argument("--beta1", default=0.9)
    parser.add_argument("--beta2", default=0.999)
    parser.add_argument("--threshold", default=0.9)
    parser.add_argument("--steps", default=8)
    parser.add_argument("--cls_epoches", default=10)
    parser.add_argument("--gan_epoches", default=5)

    opts = parser.parse_args()
    main(opts)  

