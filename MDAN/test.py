'''
test on target domain (validation data)
'''

import time
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torchvision as tv
import torchvision.models as models
from torch.utils.data import DataLoader

from model import *
from dataset import *
import Resnet

def test(model,valid_data_loader,device,mu,gamma,interval=100):

	model.eval()
	time_start = time.time()
	acc=[]
	idx=0
	# load batch data
	for x,y in tqdm(valid_data_loader):
		idx+=1
		x=x.to(device)
		y=y.to(device)
		# predict
		output=model.inference(x)
		pred = output.data.max(1)[1]  
		pred_acc = torch.sum(pred == y).item() / float(x.size(0))
		acc.append(pred_acc)
		if idx%interval==0:
			print("accuracy = %f"%(sum(acc)/len(acc)))

	accuracy=sum(acc)/len(acc)
	
	print("final accuracy = {}".format(accuracy))
	time_end = time.time()
	print("time used = {} seconds.".format(time_end - time_start))

def main(opt):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# csv_root = [os.path.join(opt.root,name,(name+'_train.csv')) for name in sorted(os.listdir(opt.root))] 
	transform = tv.transforms.Compose([tv.transforms.ToTensor(),
										tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

	valid_dataset = TestImageData(opt.root,scale=224,transforms=transform,Dt=opt.target_domain)
	valid_data_loader = DataLoader(
		valid_dataset,
		batch_size=opt.batch_size,
		shuffle=False,
		num_workers=opt.num_workers,
		drop_last=False) 
	D_name=valid_dataset.domain_name
	print('target domain',D_name)
	iterr = iter(valid_data_loader)
	x,y=iterr.next()
	print('image shape in batch:',x.shape)
	
	# load model
	model=Resnet.resnet_345()
	resnet=model.CNN
	mdan = MDANet(resnet).to(device)
	print('loading model from : ',opt.trained_model_path)
	mdan.load_state_dict(torch.load(opt.trained_model_path))

	print('Testing target domain:',D_name)
	test(mdan,valid_data_loader,device,opt.mu,opt.gamma)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
	parser.add_argument("--mu", type=float, default=0.01, help="mu")
	parser.add_argument("--gamma", type=float, default=10.0, help="gamma")
	parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
	parser.add_argument("--batch_size", type=int, default=4, help="batch size")
	parser.add_argument("--num_epochs", type=int, default=15, help="number of epochs")
	parser.add_argument("--num_domains", type=int, default=3, help="number of source domains")

	parser.add_argument("--target_domain", type=int, default=2, help="choose target domain")
	parser.add_argument("--mode", type=str, default="dynamic", help="mode of loss function")
	parser.add_argument("--root", type=str, default='./dataset_public', help="root of dataset")
	parser.add_argument("--trained_model_path", type=str, default='model/res152_MDAN013-epoch-13.pth', help="trained model path")
	opt = parser.parse_args()
	print(opt)
	main(opt)