import argparse
import torch


def save_model(checkpoint_path, model):
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)
    print ('model saved to %s' % checkpoint_path)


def load_model(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print ('model loaded from %s' % checkpoint_path)

    return model


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.sum = 0
		self.correct = 0
		self.count = 0
		self.avg = 0
		self.acc = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.correct += val
		self.count += n
		self.avg = self.sum / self.count
		self.acc = self.correct / self.count

def parse_args():
    parser = argparse.ArgumentParser(description='Muti-Source Domain Adaptation')
    # Load data
    parser.add_argument('--data_dir', dest='data_dir',
                        help='directory to dataset',
                        default="./data", type=str)
    parser.add_argument('--target', dest='target',
                        help='infograph, quickdraw, real, sketch',
                        default='real', type=str)
    parser.add_argument('--source', dest='source',
                        help='infograph, quickdraw, real, sketch',
                        default=['infograph'], type=str, nargs='+')
    parser.add_argument('--img_size', dest='img_size',
                        help='image size',
                        default=224, type=int)
    parser.add_argument('--real_test', dest='real_test',
                        help='using no label testing data of real domain',
                        default=False, type=bool)

    # Model
    parser.add_argument('--resnet50', dest='resnet50',
                        help='resnet50',
                        default=False, type=bool)
    parser.add_argument('--resnet101', dest='resnet101',
                        help='resnet101',
                        default=False, type=bool)
    parser.add_argument('--resnet152', dest='resnet152',
                        help='resnet152',
                        default=False, type=bool)
    parser.add_argument('--fixed_pretrained', dest='fixed_pretrained',
                        help='fixed pretrained cnn model',
                        default=False, type=bool)

    # Training setup
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=50, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=128, type=int)
    parser.add_argument('--manual_seed', dest='manual_seed',
                        help='manual seed',
                        default=1216, type=int)
    parser.add_argument('--train', dest='train', help='train mode',
                        action='store_true')
    parser.add_argument('--test', dest='test',
                        help='whether test while training', action='store_true')
    parser.add_argument('--early_stop', dest='early_stop',
                        help='early stopping',
                        default=5, type=int)

    # Configure optimization
    parser.add_argument('--SGD', dest='SGD',
                        help='using SGD optimizer',
                        default=False, type=bool)
    parser.add_argument('--Adam', dest='Adam',
                        help='using Adam optimizer',
                        default=False, type=bool)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-4, type=float)

    # Predict setup
    parser.add_argument('--model_path', dest='model_path',
                        help='predicted model name',
                        default='best.pth', type=str)
    parser.add_argument('--model_dir', dest='model_dir',
                        help='directory of models',
                        default='../baseline/models', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        help='predicted model name',
                        default='src_combine_tar_real_separated.pth', type=str)

    parser.add_argument('--pred_dir', dest='pred_dir',
                        help='directory of models',
                        default='', type=str)
    parser.add_argument('--title', dest='title',
                        help='directory of models',
                        default='', type=str)

    parser.add_argument('--draw_tsne', dest='draw_tsne', help='draw tsne',
                      action='store_true')

    args = parser.parse_args()

    return args
