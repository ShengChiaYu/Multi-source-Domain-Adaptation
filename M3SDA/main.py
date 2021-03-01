import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import random
import os
import sys
from os.path import join

from tqdm import tqdm, tnrange
from dataset import dataset_public
from utils import AverageMeter, parse_args, save_model, load_model
from model import resnet_345
from loss import M3SDA_Loss


def fixed_pretrained(cls):
    # fix cnn weight
    for name, child in cls.named_children():
        if name == 'CNN':
            for param in child.parameters():
                param.requires_grad = False


def resume_pretrained(resume_path, model):
    state = torch.load(resume_path)
    model.CNN.load_state_dict(state['cnn'])
    model.fc_1.load_state_dict(state['fc'])
    model.fc_2.load_state_dict(state['fc'])
    model.fc_3.load_state_dict(state['fc'])
    print ('model loaded from %s' % resume_path)

    return model


def test_model_create_output(args):
    torch.manual_seed(args.manual_seed)

    dataroot = args.data_dir
    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    target_dataset = dataset_public(root=dataroot,
                             transform=transform,
                             train=False,
                             domain=args.target,
    )
    target_loader = torch.utils.data.DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        )

    model = resnet_345(args).cuda()
    model = load_model(args.model_path, model)

    preds = []
    file_names = []
    model.eval()
    test_pbar = tqdm(total=len(target_loader), ncols=100, leave=True)
    for batch_idx, (x_inputs, y_file_names) in enumerate(target_loader):
        x_inputs = x_inputs.cuda()

        with torch.no_grad():
            output = model.predict(x_inputs)
        _, pred = torch.max(output, 1)
        preds += pred.cpu().tolist()
        file_names += y_file_names

        test_pbar.update()

    test_pbar.close()

    with open(join(args.pred_dir, 'output_{}.csv'.format(args.target)), 'w') as f:
        f.write('image_name,label\n')
        for i, pred in enumerate(preds):
            f.write('{}/{},{}\n'.format(args.title, file_names[i], pred))


def test_model(args):
    torch.manual_seed(args.manual_seed)

    dataroot = join(os.getcwd(), args.data_dir)
    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    target_dataset = dataset_public(root=dataroot,
                             transform=transform,
                             train=False,
                             domain=args.target,
    )
    target_loader = torch.utils.data.DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        )

    model = resnet_345(args).cuda()
    model = load_model(args.model_path, model)

    preds = []
    labels = []
    model.eval()
    test_pbar = tqdm(total=len(target_loader), ncols=100, leave=True)
    for batch_idx, (x_inputs, y_labels) in enumerate(target_loader):
        x_inputs = x_inputs.cuda()

        with torch.no_grad():
            output = model.predict(x_inputs)
        _, pred = torch.max(output, 1)
        preds += pred.cpu().tolist()
        labels += y_labels.tolist()

        test_pbar.update()

    test_pbar.close()

    preds = np.array(preds)
    labels = np.array(labels)
    acc = float(sum(preds == labels)) / float(len(preds))
    print ('valid acc = {:4f}'.format(acc))

    return acc



def train_model(args):
    torch.manual_seed(args.manual_seed)

    dataroot = join(os.getcwd(), args.data_dir)
    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    source_dataset, source_loader = [], []
    for source in args.source:
        dataset = dataset_public(root=dataroot,
                                 transform=transform,
                                 train=True,
                                 domain=source,
        )
        loader = torch.utils.data.DataLoader(
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
    target_loader = torch.utils.data.DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        )


    # models
    cls = resnet_345(args).cuda()

    resume_path = join(os.getcwd(), args.model_dir, args.model_name)
    resume_pretrained(resume_path, cls)

    if args.fixed_pretrained:
        fixed_pretrained(cls)

    # print('Training parameters')
    params_to_update = []
    for name, param in cls.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            # print(name)
    print('Number of parameters to update: {}'.format(len(params_to_update)))

    # optimizer
    if args.SGD:
        optimizer = optim.SGD(params_to_update,  lr=args.lr, momentum=0.9)
    elif args.Adam:
        optimizer = optim.Adam(params_to_update, lr=args.lr)

    best_acc = 0.0
    patience = 0
    f = open(join('./logs', 'log_{}.txt'.format(args.target)), 'w')
    f.write('cls_loss,md_loss,acc\n')
    for epoch in range(args.start_epoch, args.max_epochs+1):
        criterion = M3SDA_Loss()
        cls.train()

        print ('\nEpoch = {}'.format(epoch))

        cls_losses, md_losses, train_acc = AverageMeter(), AverageMeter(), AverageMeter()
        len_train_loader = min([len(loader) for loader in (source_loader + [target_loader])])
        train_pbar = tqdm(total=len_train_loader, ncols=100, leave=True)
        for i, batch_data in enumerate(zip(source_loader[0], source_loader[1], source_loader[2], target_loader)):
            data_1, data_2, data_3, data_t = batch_data
            input_1, label_1 = data_1
            input_2, label_2 = data_2
            input_3, label_3 = data_3
            input_t, label_t = data_t

            input_1, input_2, input_3, input_t = input_1.cuda(), input_2.cuda(), input_3.cuda(), input_t.cuda()
            label_1, label_2, label_3, label_t = label_1.cuda(), label_2.cuda(), label_3.cuda(), label_t.cuda()

            cls.zero_grad()

            e1, e2, e3, et, out_1, out_2, out_3 = cls(input_1, input_2, input_3, input_t)
            cls_loss, md_loss = criterion(e1, e2, e3, et, out_1, out_2, out_3, label_1, label_2, label_3)
            loss = cls_loss + md_loss
            loss.backward()
            optimizer.step()

            output = torch.cat((out_1, out_2, out_3), 0)
            labels = torch.cat((label_1, label_2, label_3))
            _, pred = torch.max(output, 1)
            acc = sum(pred == labels).float() / float(pred.shape[0])

            cls_losses.update(cls_loss.data.item(), input_1.shape[0])
            md_losses.update(md_loss.data.item(), input_1.shape[0])
            train_acc.update(acc, 1)

            train_pbar.update()
            train_pbar.set_postfix({'cls_loss':'{:.4f}'.format(cls_losses.avg), 'md_loss':'{:.4f}'.format(md_losses.avg), 'acc':'{:.4f}'.format(train_acc.avg)})

        train_pbar.close()
        if epoch % 1 == 0:
            acc, _ = eval_model(cls, target_loader)
            f.write('{:4f},{:4f},{:4f}\n'.format(cls_losses.avg, md_losses.avg, acc))
            f.flush()
            if acc > best_acc:
                best_acc = acc
                save_model(join('./models', 'm3sda_{}_{}_{}.pth'.format(args.target, epoch, acc)), cls)
                patience = 0
            else:
                patience += 1

            if patience >= args.early_stop:
                print ('early stopping...')
                break


def eval_model(model, target_loader):
    preds = []
    labels = []
    model.eval()
    eval_pbar = tqdm(total=len(target_loader), ncols=100, leave=True)
    for batch_idx, (x_inputs, y_labels) in enumerate(target_loader):
        x_inputs, y_labels = x_inputs.cuda(), y_labels.cuda()

        with torch.no_grad():
            output = model.predict(x_inputs)
        _, pred = torch.max(output, 1)
        preds += pred.cpu().tolist()
        labels += y_labels.cpu().tolist()

        eval_pbar.update()
        if batch_idx == int(len(target_loader) / 5): break  # save time

    eval_pbar.close()

    preds = np.array(preds)
    labels = np.array(labels)
    acc = float(sum(preds == labels)) / float(len(preds))
    print ('valid acc = {:4f}'.format(acc))

    return acc, preds


def draw_tsne(model, source_testloader, target_testloader, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.manifold import TSNE

    model.eval()

    def get_features(target_testloader, k):
        t_features, t_labels = [], []
        test_pbar = tqdm(total=len(target_testloader), ncols=100, leave=True)
        for batch_idx, (t_input, t_label) in enumerate(target_testloader):
            t_input, t_label = t_input.cuda(), t_label.cuda()

            with torch.no_grad():
                feature = model.CNN(t_input)
                t_features += feature.cpu().tolist()
                t_labels += t_label.cpu().tolist()

            test_pbar.update()
            if batch_idx * 128 > k:
                break
        test_pbar.close()

        return t_features[:k], t_labels[:k]

    k = 2000
    s_1_features, s_1_labels = get_features(source_testloader[0], k)
    s_2_features, s_2_labels = get_features(source_testloader[1], k)
    s_3_features, s_3_labels = get_features(source_testloader[2], k)
    t_features, t_labels = get_features(target_testloader, k)

    all_features = np.array(s_1_features + s_2_features + s_3_features + t_features)
    all_domains = np.array([0] * len(s_1_features) + [1] * len(s_2_features) + [2] * len(s_3_features)+ [3] * len(t_features))
    all_labels = np.array(s_1_labels + s_2_labels + s_3_labels + t_labels)

    # plot digits
    #all_indices = all_labels < 20
    #all_features = all_features[all_indices]
    #all_labels = all_labels[all_indices]

    tsne = TSNE(init='pca')
    all_features = tsne.fit_transform(all_features)

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

    #legends = []
    #for label in range(10):
    #    legends.append(mpatches.Patch(color=color_list[label], label='{}'.format(label)))
    '''
    size = 5
    for idx, feature in enumerate(all_features):
        label = all_labels[idx]
        plt.scatter(feature[0], feature[1], c=color_list[label], s=size)

    #plt.legend(handles=legends)
    plt.savefig('digits_{}.png'.format('real'), bbox_inches='tight')
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
    plt.savefig('domain_all.png', bbox_inches='tight')
    plt.gcf().clear()



if __name__ == '__main__':
    args = parse_args()

    if args.train:
        train_model(args)

    if args.test:
        test_model_create_output(args)

    if args.draw_tsne:
        # generate dataloader
        dataroot = join(os.getcwd(), args.data_dir)
        transform=transforms.Compose([
                  transforms.Resize((args.img_size,args.img_size)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        source_dataset, source_loader = [], []
        for source in args.source:
            dataset = dataset_public(root=dataroot,
                                     transform=transform,
                                     train=False,
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

        model = resnet_345(args).cuda()
        model = load_model(args.model_path, model)

        draw_tsne(model, source_loader, target_loader, args)
