
'''
train MDAN with fine-tuned resnet152 on 3 source domains and a target domain
all domains
0 = infograph 
1 = quickdraw
2 = real (target domain on kaggle)
3 = sketch

reference: https://github.com/KeiraZhao/MDAN
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


def train(model,Ds,Dt,D_name,data_loader,device,opt,interval=10):
	print('Source domain:',[D_name[i] for i in Ds])
	print('Target domain:',D_name[Dt])
	# optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
	optimizer = optim.SGD(model.parameters(), lr=opt.lr,momentum=0.9)
	model.train()
	time_start = time.time()

	for ep in range(opt.num_epochs):
		running_loss = 0.0
		b=0	
		b_loss=0
		Acc=[]
		# load batch data
		for (x,y) in tqdm(data_loader):
			b+=1
			xs=[]
			ys=[]
			for i in Ds:
				xs.append(x[i].to(device))
				ys.append(y[i].to(device))
			xt=x[Dt].to(device)
			
			# define domain lables
			slabels = torch.ones(xt.shape[0], requires_grad=False).type(torch.LongTensor).to(device)
			tlabels = torch.zeros(xt.shape[0], requires_grad=False).type(torch.LongTensor).to(device)
			
			optimizer.zero_grad()

			# predict
			logprobs, sdomains, tdomains = model(xs, xt)
			# print(logprobs)
			# Compute prediction accuracy on multiple training sources.
			losses = torch.stack([F.nll_loss(logprobs[j], ys[j]) for j in range(opt.num_domains)])
			domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
									F.nll_loss(tdomains[j], tlabels) for j in range(opt.num_domains)])
			
			# Different final loss function depending on different training modes.
			if opt.mode == "maxmin":
				loss = torch.max(losses) + opt.mu * torch.min(domain_losses)
			elif opt.mode == "dynamic":
				loss = torch.log(torch.sum(torch.exp(opt.gamma * (losses + opt.mu * domain_losses)))) / opt.gamma
			elif opt.mode == "soft":
				loss = torch.log(torch.sum(torch.exp(opt.gamma * (losses + torch.min(domain_losses))))) / opt.gamma
			else:
				raise ValueError("No support for the training mode on madnNet: {}.".format(opt.mode))
			# print("Batch {}/{}, loss = {}".format(b,len(data_loader), loss.item()))	
			running_loss += loss.item()			
			b_loss+=loss.item()
			loss.backward()
			optimizer.step()

			model.eval()
			yt=y[Dt].to(device)
			output=model.inference(xt)
			pred = output.data.max(1, keepdim=True)[1]  
			pred_acc = torch.sum(pred == yt).item() / float(xt.size(0))
			Acc.append(pred_acc)
			
			if b%interval==0:			
				
				# print("Batch {},  accuracy = {}".format(b ,pred_acc))
				print("Batch {}, loss = {},accuracy = {}".format(b, b_loss/interval,sum(Acc)/len(Acc)))
				b_loss=0
			model.train()

			
		print("Iteration {}, loss = {}, accuracy = {}".format(ep, running_loss/b,sum(Acc)/len(Acc)))

		torch.save(model.state_dict(), os.path.join(opt.model_path,'res152_MDAN-{}-epoch-{}.pth'.format(D_name[Dt],ep+46)))
	time_end = time.time()
	torch.save(model.state_dict(), os.path.join(opt.model_path,'res152_MDAN-{}-epoch-{}.pth'.format(D_name[Dt],ep+46)))
	print("time used = {} seconds.".format(time_end - time_start))



def main(opt):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# csv_root = [os.path.join(opt.root,name,(name+'_train.csv')) for name in sorted(os.listdir(opt.root))] 
	transform = tv.transforms.Compose([tv.transforms.ToTensor(),
										tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

	train_dataset = WholeImageData(opt.root,scale=224,transforms=transform)
	train_data_loader = DataLoader(
		train_dataset,
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=opt.num_workers,
		drop_last=False) 
		
	D_name=train_dataset.domain_name
	iterr = iter(train_data_loader)
	x,_=iterr.next()
	print('image shape in batch:',x[0].shape)

	# # Train DannNet.
	# # resnet,fc = get_feature_ex(model_path=opt.pretrained_model_path)
	# # resnet,fc=resnet.to(device),fc.to(device)
	# resnet101=models.resnet101(pretrained=True)
	# resnet=Net(resnet101).to(device)
	# mdan = MDANet(resnet).to(device)
	# # mdan.softmax=fc

	# load model
	# model=Resnet.resnet_345()
	# resnet=model.CNN
	# mdan = MDANet(resnet).to(device)

	resnet101=models.resnet101(pretrained=True)
	resnet=Net(resnet101).to(device)
	mdan = MDANet(resnet).to(device)
	print('loading model from : soft_model/res152_MDAN-real-epoch-45.pth')
	mdan.load_state_dict(torch.load('soft_model/res152_MDAN-real-epoch-45.pth'))

	Ds,Dt=choose_domain(opt.target_domain)
	train(mdan,Ds,Dt,D_name=D_name,data_loader=train_data_loader,device=device,opt=opt)

def get_feature_ex(model_path='resnet152_real.pth'):
	model=Resnet.resnet_345()
	# state = torch.load(model_path)
	# model.load_state_dict(state['state_dict'])
	resnet=model.CNN
	fc=model.fc
	return resnet,fc

def choose_domain(Dt):
	Ds=[]
	for i in range(4):
		if i!=Dt:
			Ds.append(i)
	return Ds,Dt
	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
	parser.add_argument("--mu", type=float, default=0.01, help="mu")
	parser.add_argument("--gamma", type=float, default=10.0, help="gamma")
	parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
	parser.add_argument("--batch_size", type=int, default=8, help="batch size")
	parser.add_argument("--num_epochs", type=int, default=60, help="number of epochs")
	parser.add_argument("--num_domains", type=int, default=3, help="number of source domains")

	parser.add_argument("--target_domain", type=int, default=2, help="choose target domain")
	parser.add_argument("--mode", type=str, default="dynamic", help="mode of loss function")
	parser.add_argument("--root", type=str, default='./dataset_public', help="root of dataset")
	parser.add_argument("--model_path", type=str, default='model', help="save model path")
	parser.add_argument("--pretrained_model_path", type=str, default='resnet152_real.pth', help="source combined model path")
	opt = parser.parse_args()
	print(opt)
	main(opt)
