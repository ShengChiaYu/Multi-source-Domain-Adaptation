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

def infer(model,test_data_loader,opts,device):

	# if not os.path.exists(opts.model_dir):
	# 	os.makedirs(opts.model_dir, exist_ok=True)

	model.eval()
	n_pred = []
	n_name = []
	for img,img_name in tqdm(test_data_loader):
		with torch.no_grad():
			img = img.to(device)

			output=model.inference(img)
			pred = output.data.max(1, keepdim=True)[1]  

			for v_pred in pred:
				n_pred.append(v_pred.item())

			for v_name in img_name:
				n_name.append(v_name)

	output_path = opts.pred_dir
	print('=====Write output to %s =====' % output_path)
	
	with open(output_path, 'w') as f:
		f.write('image_name,label\n')
		for i, (s_pred,s_name) in enumerate(zip(n_pred,n_name)): 
			f.write('{},{}\n'.format(s_name,s_pred))
	

def main(opt):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	transform = tv.transforms.Compose([tv.transforms.ToTensor(),
										tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


	test_dataset     = ImageDataNoLabel(opt.root,opt.test_root,scale=224,transforms=transform)
	test_data_loader = DataLoader(
		test_dataset,
		batch_size=opt.batch_size,
		shuffle=False,
		num_workers=opt.num_workers,
		drop_last=False) 

	iterr = iter(test_data_loader)
	x,_=iterr.next()
	print('image shape in batch:',x[0].shape)

	# load model
	model=Resnet.resnet_345()
	resnet=model.CNN
	mdan = MDANet(resnet).to(device)
	print('loading model from : ',opt.trained_model_path)
	mdan.load_state_dict(torch.load(opt.trained_model_path))


	infer(model=mdan,test_data_loader=test_data_loader,opts=opt,device=device)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
	parser.add_argument("--batch_size", type=int, default=2, help="batch size")
	parser.add_argument("--root", type=str, default='./dataset_public', help="Path to dataset")
	parser.add_argument('--test_root',type = str,default = 'test', help = 'Path to test_root') 
	parser.add_argument("--trained_model_path", type=str, default='model/res152_MDAN013-epoch-13.pth', help="read model path")
	parser.add_argument('--pred_dir',type=str,default='pred.csv',help= 'Path to save the pred_dir')
	opt = parser.parse_args()
	
	print(opt)
	main(opt)
