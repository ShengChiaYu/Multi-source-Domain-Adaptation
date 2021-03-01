import argparse

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.sum = 0
		self.count = 0
		self.avg = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def parse_args():
    parser = argparse.ArgumentParser(description='Muti-Source Domain Adaptation')
    # Load data
    parser.add_argument('--data_dir', dest='data_dir',
                        help='directory to dataset',
                        default="../dataset_public", type=str)
    parser.add_argument('--target', dest='target',
                        help='infograph, quickdraw, real, sketch',
                        default=['real'], type=str, nargs='+')
    parser.add_argument('--source', dest='source',
                        help='infograph, quickdraw, real, sketch',
                        default=['infograph'], type=str, nargs='+')
    parser.add_argument('--img_size', dest='img_size',
                        help='image size',
                        default=224, type=int)
    parser.add_argument('--real_test', dest='real_test',
                        help='using no label testing data of real domain',
                        default=False, action='store_true')

    # Model
    parser.add_argument('--resnet50', dest='resnet50',
                        help='resnet50',
                        default=False, action='store_true')
    parser.add_argument('--resnet101', dest='resnet101',
                        help='resnet101',
                        default=False, action='store_true')
    parser.add_argument('--resnet152', dest='resnet152',
                        help='resnet152',
                        default=False, action='store_true')
    parser.add_argument('--fixed_pretrained', dest='fixed_pretrained',
                        help='fixed pretrained cnn model',
                        default=False, action='store_true')
    parser.add_argument('--extractor', dest='extractor',
                        help='cnn model',
                        default=False, action='store_true')

    # Training setup
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=50, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=6, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=32, type=int)
    parser.add_argument('--manual_seed', dest='manual_seed',
                        help='manual seed',
                        default=1216, type=int)
    parser.add_argument('--test', dest='test',
                        help='whether test while training',
                        default=False, action='store_true')
    parser.add_argument('--early_stopping', dest='early_stopping',
                        help='early stopping',
                        default=5, type=int)

    # Configure optimization
    parser.add_argument('--SGD', dest='SGD',
                        help='using SGD optimizer',
                        default=False, action='store_true')
    parser.add_argument('--Adam', dest='Adam',
                        help='using Adam optimizer',
                        default=False, action='store_true')
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-3, type=float)

    # Predict setup
    parser.add_argument('--model_dir', dest='model_dir',
                        help='directory of models',
                        default='../baseline/models', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        help='predicted model name',
                        default='src_combine_tar_real_separated.pth', type=str)

    args = parser.parse_args()

    return args
