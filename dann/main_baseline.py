import os
import sys
import argparse
from os.path import join
import pickle

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import dataset_public
from models import DANN, resnet_345
from utils import AverageMeter

np.random.seed(1124)

self_acc_list = [0.3537, 0.6827, 0.6629]
real_acc_lsit = [0.4245, 0.0645, 0.4693]


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DANN network')
    parser.add_argument('--train', dest='train', help='train mode',
                      action='store_true')
    parser.add_argument('--model', dest='model',
                        help='model',
                        default='resnet152', type=str)

    parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)

    parser.add_argument('--target', dest='target',
                        help='infograph, quickdraw, real, sketch',
                        default='real', type=str)
    parser.add_argument('--source', dest='source',
                        help='infograph, quickdraw, real, sketch',
                        default=['infograph'], type=str, nargs='+')
    parser.add_argument('--img_size', dest='img_size',
                        help='image size',
                        default=224, type=int)

    parser.add_argument('--data_dir', dest='data_dir',
                      help='directory to dataset', default="../data",
                      type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
    parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
    parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=7, type=int)
    parser.add_argument('--early_stop', dest='early_stop',
                      help='early_stop',
                      default=5, type=int)
    parser.add_argument('--cuda', dest='use_cuda', help='whether use CUDA',
                      default=False, type=bool)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-4, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not', action='store_true')
    parser.add_argument('--resume_path', dest='resume_path',
                      help='checkpoint to load model',
                      default='models/pretrained/resnet152_real_separated.pth', type=str)

    # evaluation
    parser.add_argument('--test', dest='test', help='test mode',
                      action='store_true')
    parser.add_argument('--test_dir', dest='test_dir',
                      help='directory to testing dataset',
                      default="hw2_train_val/val1500", type=str)
    parser.add_argument('--load_model', dest='load_model_path',
                      help='filepath to load model',
                      default="models/model_30.pth", type=str)
    parser.add_argument('--output_csv', dest='output_csv',
                      help='filepath to predict output',
                      default="test_pred.csv", type=str)

    parser.add_argument('--draw_tsne', dest='draw_tsne', help='draw tsne',
                      action='store_true')

    args = parser.parse_args()
    return args


def save_checkpoint(checkpoint_path, model, optimizer, epoch):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict(),
             'epoch' : epoch}
    torch.save(state, checkpoint_path)
    print ('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    epoch = state['epoch']
    print ('model loaded from %s' % checkpoint_path)

    return model, optimizer, epoch + 1


def resume_pretrained(resume_path, model):
    state = torch.load(resume_path)
    model.feature_extractor.load_state_dict(state['cnn'])
    model.class_classifier.load_state_dict(state['fc'])
    print ('model loaded from %s' % resume_path)

    return model


def train(model, source_loader, target_loader, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    if args.resume == True:
        model = resume_pretrained(args.resume_path, model)


    f = open('dann_result_{}.txt'.format(args.target), 'w')
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # training
        print ('Epoch = {}'.format(epoch))
        model.train()
        losses, class_losses, domain_losses, train_acc, domain_acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        len_train_loader = min([len(loader) for loader in (source_loader + [target_loader])])
        train_pbar = tqdm(total=len_train_loader, ncols=100, leave=True)

        for batch_idx, batch_data in enumerate(zip(source_loader[0], source_loader[1], source_loader[2], target_loader)):
            data_1, data_2, data_3, data_t = batch_data
            input_1, label_1 = data_1
            input_2, label_2 = data_2
            input_3, label_3 = data_3
            input_t, label_t = data_t

            domain_1_label = torch.zeros(input_1.shape[0]).fill_(0).long()
            domain_2_label = torch.zeros(input_1.shape[0]).fill_(0).long()
            domain_3_label = torch.zeros(input_1.shape[0]).fill_(0).long()
            domain_t_label = torch.zeros(input_1.shape[0]).fill_(1).long()

            if args.use_cuda:
                input_1, input_2, input_3, input_t = input_1.cuda(), input_2.cuda(), input_3.cuda(), input_t.cuda()
                label_1, label_2, label_3, label_t = label_1.cuda(), label_2.cuda(), label_3.cuda(), label_t.cuda()
                domain_1_label, domain_2_label, domain_3_label, domain_t_label = domain_1_label.cuda(), domain_2_label.cuda(), domain_3_label.cuda(), domain_t_label.cuda()

            # calculate alpha
            p = float(batch_idx + epoch * len_train_loader) / args.max_epochs / len_train_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # source dataset
            class_1_output, domain_1_output = model(input_1, alpha=alpha)
            class_1_loss = criterion(class_1_output, label_1)
            domain_1_loss = criterion(domain_1_output, domain_1_label)

            class_2_output, domain_2_output = model(input_2, alpha=alpha)
            class_2_loss = criterion(class_2_output, label_2)
            domain_2_loss = criterion(domain_2_output, domain_2_label)

            class_3_output, domain_3_output = model(input_3, alpha=alpha)
            class_3_loss = criterion(class_3_output, label_3)
            domain_3_loss = criterion(domain_3_output, domain_3_label)

            # target dataset
            class_t_output, domain_t_output = model(input_t, alpha=alpha)
            domain_t_loss = criterion(domain_t_output, domain_t_label)

            # collect losses
            class_loss = class_1_loss + class_2_loss + class_3_loss
            domain_loss = domain_1_loss + domain_2_loss + domain_3_loss + domain_t_loss
            loss = class_loss + domain_loss
            losses.update(loss.data.item(), input_1.shape[0])
            class_losses.update(class_loss.data.item(), input_1.shape[0])
            domain_losses.update(domain_loss.data.item(), input_1.shape[0])

            # calculate accuracy
            output = torch.cat((class_1_output, class_2_output, class_3_output), 0)
            labels = torch.cat((label_1, label_2, label_3))
            _, pred = torch.max(output, 1)
            acc = sum(pred == labels).float() / float(pred.shape[0])
            train_acc.update(acc, 1)

            d_output = torch.cat((domain_1_output, domain_2_output, domain_3_output, domain_t_output), 0)
            d_labels = torch.cat((domain_1_label, domain_2_label, domain_3_label, domain_t_label))
            _, d_pred = torch.max(d_output, 1)
            d_acc = sum(d_pred == d_labels).float() / float(d_pred.shape[0])
            domain_acc.update(d_acc, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pbar.update()

            train_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg), 'class':'{:.4f}'.format(class_losses.avg),\
                                    'domain':'{:.4f}'.format(domain_losses.avg), 'acc':'{:.4f}'.format(train_acc.avg), 'd_acc':'{:.4f}'.format(domain_acc.avg)})
        train_pbar.close()

        # validation on target dataset
        acc = test(model, target_loader, args)
        f.write('{:.4f}\n'.format(acc))
        if acc > best_acc:
            best_acc = acc
            patience = 0
            save_checkpoint(join(args.save_dir, '2d_{}_{}_{:.6f}.pth'.format(args.target, epoch, acc)), model, optimizer, epoch)
        else:
            patience += 1

        if patience >= args.early_stop:
            print ('early stopping...')
            break




def test(model, target_loader, args):
    predictions = []
    labels = []
    model.eval()
    test_pbar = tqdm(total=len(target_loader), ncols=100, leave=True)
    for batch_idx, (t_input, t_label) in enumerate(target_loader):
        if args.use_cuda:
            t_input, t_label = t_input.cuda(), t_label.cuda()

        with torch.no_grad():
            class_output = model(t_input, alpha=0, pred_only=True)
            prediction = class_output.data.max(1, keepdim=True)[1].squeeze()
            predictions += prediction.cpu().tolist()
            labels += t_label.cpu().tolist()

        test_pbar.update()
        if batch_idx == int(len(target_loader) / 5): break  # save time
    test_pbar.close()

    acc = np.sum(np.array(predictions) == np.array(labels)) / float(len(predictions))
    print ('test set acc = {}'.format(acc))

    return acc


def test_model_create_output(model, args, target_loader):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model, _, _ = load_checkpoint(args.load_model_path, model, optimizer)

    predictions = []
    file_names = []
    model.eval()
    test_pbar = tqdm(total=len(target_loader), ncols=100, leave=True)
    for batch_idx, (t_input, y_file_names) in enumerate(target_loader):
        if args.use_cuda:
            t_input = t_input.cuda()

        with torch.no_grad():
            class_output = model(t_input, alpha=0, pred_only=True)
            prediction = class_output.data.max(1, keepdim=True)[1].squeeze()
            predictions += prediction.cpu().tolist()
            file_names += y_file_names

        test_pbar.update()
    test_pbar.close()

    with open('output_{}.csv'.format(args.target), 'w') as f:
        f.write('image_name,label\n')
        for i, pred in enumerate(predictions):
            f.write('test/{},{}\n'.format(file_names[i], pred))


def test_model(model, target_loader, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model, _, _ = load_checkpoint(args.load_model_path, model, optimizer)

    predictions = []
    labels = []
    model.eval()
    test_pbar = tqdm(total=len(target_loader), ncols=100, leave=True)
    for batch_idx, (t_input, t_label) in enumerate(target_loader):
        if args.use_cuda:
            t_input, t_label = t_input.cuda(), t_label.cuda()

        with torch.no_grad():
            class_output = model(t_input, alpha=0, pred_only=True)
            prediction = class_output.data.max(1, keepdim=True)[1].squeeze()
            predictions += prediction.cpu().tolist()
            labels += t_label.cpu().tolist()

        test_pbar.update()
        if batch_idx == int(len(target_loader) / 5): break  # save time
    test_pbar.close()

    acc = np.sum(np.array(predictions) == np.array(labels)) / float(len(predictions))
    print ('test set acc = {}'.format(acc))

    return acc


def draw_tsne(model, source_testloader, target_testloader, args):
    try:
        with open('real_encoding.pkl', 'rb') as f:
            pkl = pickle.load(f)
            all_features = pkl['all_features']
            all_domains = pkl['all_domains']
            all_labels = pkl['all_labels']

    except:
        model = resume_pretrained(args.resume_path, model)
        model.eval()

        def get_features(target_testloader, k):
            t_features, t_labels = [], []
            test_pbar = tqdm(total=len(target_testloader), ncols=100, leave=True)
            for batch_idx, (t_input, t_label) in enumerate(target_testloader):
                if args.use_cuda:
                    t_input, t_label = t_input.cuda(), t_label.cuda()

                with torch.no_grad():
                    feature = model.feature_extractor(t_input)
                    t_features += feature.cpu().tolist()
                    t_labels += t_label.cpu().tolist()

                test_pbar.update()
                if batch_idx * 128 > k:
                    break
            test_pbar.close()

            return t_features[:k], t_labels[:k]

        k = 1500
        s_1_features, s_1_labels = get_features(source_testloader[0], k)
        s_2_features, s_2_labels = get_features(source_testloader[1], k)
        s_3_features, s_3_labels = get_features(source_testloader[2], k)
        t_features, t_labels = get_features(target_testloader, k)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from sklearn.manifold import TSNE

        def generate_color_list():
            color_list = []
            for i in range(7):
                color_list.append(plt.cm.Set2(i))

            for i in range(8):
                color_list.append(plt.cm.Set1(i))

            for i in range(10):
                color_list.append(plt.cm.Set3(i))

            return color_list

        color_list = generate_color_list()

        all_features = np.array(s_1_features + s_2_features + s_3_features + t_features)
        all_domains = np.array([0] * len(s_1_features) + [1] * len(s_2_features) + [2] * len(s_3_features)+ [3] * len(t_features))
        all_labels = np.array(s_1_labels + s_2_labels + s_3_labels + t_labels)

        tsne = TSNE(init='pca')
        all_features = tsne.fit_transform(all_features)

        data = {'all_features': all_features, 'all_domains': all_domains, 'all_labels': all_labels}
        with open('real_encoding.pkl', 'wb') as f:
            pickle.dump(data, f)

    # plot digits
    '''
    legends = []
    for label in range(10):
        legends.append(mpatches.Patch(color=color_list[label], label='{}'.format(label)))

    for idx, feature in enumerate(all_features):
        label = all_labels[idx]
        plt.scatter(feature[0], feature[1], c=color_list[label], s=size)

    plt.legend(handles=legends)
    plt.savefig('digits_{}_{}.png'.format(args.source_domain, args.target_domain), bbox_inches='tight')
    plt.gcf().clear()
    '''

    # plot domains
    legends = []
    legends.append(mpatches.Patch(color=color_list[0], label='{}'.format(args.source[0])))
    legends.append(mpatches.Patch(color=color_list[1], label='{}'.format(args.source[1])))
    legends.append(mpatches.Patch(color=color_list[2], label='{}'.format(args.source[2])))
    legends.append(mpatches.Patch(color=color_list[3], label='{}'.format(args.target)))

    size = 5
    for idx, feature in enumerate(all_features):
        label = all_domains[idx]
        plt.scatter(feature[0], feature[1], c=color_list[label], s=size)

    plt.legend(handles=legends)
    plt.savefig('domain_{}.png'.format(args.target), bbox_inches='tight')
    plt.gcf().clear()


if __name__ == '__main__':
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()

    # create dataloader
    dataroot = join(os.getcwd(), args.data_dir)
    transform=transforms.Compose([
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # create YOLO model
    model = resnet_345()
    if args.use_cuda:
        model = model.cuda()

    checkpoint_path = 'models/baseline/src_{}.pth'.format(args.target)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

    target_dataset = dataset_public(root=dataroot,
                             transform=transform,
                             train=False,
                             domain='real',
    )
    target_loader = DataLoader(
        dataset=target_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        )

    predictions = []
    labels = []
    vectors = []
    model.eval()
    test_pbar = tqdm(total=len(target_loader), ncols=100, leave=True)
    for batch_idx, (t_input, t_label) in enumerate(target_loader):
        if args.use_cuda:
            t_input = t_input.cuda()

        with torch.no_grad():
            class_output = model(t_input)
            vectors += class_output.cpu().tolist()

            #prediction = class_output.data.max(1, keepdim=True)[1].squeeze()
            #predictions += prediction.cpu().tolist()
            #labels += t_label.cpu().tolist()

        test_pbar.update()
        #if batch_idx == 5: break  # save time
    test_pbar.close()

    #acc = np.sum(np.array(predictions) == np.array(labels)) / float(len(predictions))
    #print ('test set acc = {}'.format(acc))

    vectors = np.array(vectors)
    print (vectors.shape)
    #mean_vectors = np.mean(vectors, axis=0)
    #print (mean_vectors.shape)

    data = {'vectors': vectors}
    with open('real_vectors_using_{}.pkl'.format(args.target), 'wb') as f:
        pickle.dump(data, f)



    if args.train:
        # generate dataloader
        source_dataset, source_loader = [], []
        for source in args.source:
            dataset = dataset_public(root=dataroot,
                                     transform=transform,
                                     train=True,
                                     domain=source,
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                )

            source_dataset.append(dataset)
            source_loader.append(loader)

        target_dataset = dataset_public(root=dataroot,
                                 transform=transform,
                                 train=True,
                                 domain=args.target,
        )
        target_loader = DataLoader(
            dataset=target_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            )

    if args.test:
        target_dataset = dataset_public(root=dataroot,
                                 transform=transform,
                                 train=False,
                                 domain=args.target,
        )
        target_loader = torch.utils.data.DataLoader(
            dataset=target_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            )

    # start training
    if args.train:
        train(model, source_loader, target_loader, args)

    if args.test:
        if args.target == 'real':
            test_model_create_output(model, args, target_loader)
        else:
            test_model(model, target_loader, args)

    if args.draw_tsne:
        # generate dataloader
        source_dataset, source_loader = [], []
        for source in args.source:
            dataset = dataset_public(root=dataroot,
                                     transform=transform,
                                     train=True,
                                     domain=source,
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=128,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                )

            source_dataset.append(dataset)
            source_loader.append(loader)

        target_dataset = dataset_public(root=dataroot,
                                 transform=transform,
                                 train=True,
                                 domain=args.target,
        )
        target_loader = DataLoader(
            dataset=target_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            )

        draw_tsne(model, source_loader, target_loader, args)
