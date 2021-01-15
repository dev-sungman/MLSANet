import os
import sys
import random
import numpy as np
from config import parse_arguments
from datasets import ClassPairDataset
from utils import register_forward_hook, visualize_activation_map
from models.acm_resnet import acm_resnet50, acm_resnet152

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.modules.distance import PairwiseDistance

from tensorboardX import SummaryWriter

import time
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

def ACM_loss(logit):
    return 2-(2*logit)

def train(args, data_loader, test_loader, model, optimizer, scheduler, device, writer, log_dir, checkpoint_dir):
    
    model.train()
    correct = 0
    total = 0
    
    overall_iter = args.start_iter
    img_dir = os.path.join(log_dir, 'imgs')
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        # NEED TO UPDATE DATSETS. 
        iter_ = 0 
        running_loss = 0
        running_change = 0
        running_disease = 0

        for base, fu, change_labels, disease_labels in iter(data_loader):
            base = base.to(device)
            fu = fu.to(device)

            change_labels = change_labels.to(device)
            disease_labels = [disease_labels[0].to(device), disease_labels[1].to(device)]

            # for activation
            activation, layer_names = register_forward_hook(model)

            base_embed, fu_embed, outputs = model(base,fu)
            
            _, preds = outputs.max(1)
            total += change_labels.size(0)
            correct += preds.eq(change_labels).sum().item()
            
            preds_cpu = preds.cpu().numpy().tolist()
            labels_cpu = change_labels.cpu().numpy().tolist()
            
            # change loss
            ce_criterion = nn.CrossEntropyLoss()
            change_loss = ce_criterion(outputs, change_labels)

            # disease loss
            disease_loss = ce_criterion(base_embed, disease_labels[0]) + ce_criterion(fu_embed, disease_labels[1])
            overall_loss = change_loss + disease_loss
            
            running_change += change_loss.item()
            running_disease += disease_loss.item()
            running_loss += overall_loss.item()

            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()
            
            if (iter_ % args.print_freq == 0) & (iter_ != 0):
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                print('Epoch: {:2d}, LR: {:5f}, Iter: {:5d}, Change loss: {:5f}, Disease loss: {:5f}, Overall loss: {:5f}, Acc: {:4f}'.format(epoch, lr, iter_, running_change/iter_, running_disease/iter_, running_loss/iter_, 100.*correct/total))
                writer.add_scalar('change_loss', running_change/iter_, overall_iter)
                writer.add_scalar('disease_loss', running_disease/iter_, overall_iter)
                writer.add_scalar('train_acc', 100.*correct/total, overall_iter)

            iter_ += 1
            overall_iter += 1
        
        scheduler.step()
        test(args, test_loader, model, device, writer, log_dir, checkpoint_dir, overall_iter)
        torch.save({
            'epoch' : epoch + 1,
            'iter' : overall_iter,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, str(overall_iter)) + '.pth')


def test(args, data_loader, model, device, writer, log_dir, checkpoint_dir, iter_):
    print('[*] Test Phase')
    model.eval()
    
    correct = 0
    total = 0
    img_dir = os.path.join(log_dir, 'imgs')
    
    with torch.no_grad():
        for base, fu, change_labels, disease_labels in iter(data_loader):
            base = base.to(device)
            fu = fu.to(device)
            change_labels = change_labels.to(device)
            
            activation, layer_names = register_forward_hook(model)
            _, _, outputs = model(base,fu)

            _, preds = outputs.max(1)
            preds_cpu = preds.cpu().numpy().tolist()

            ### Change / No-change
            total += change_labels.size(0)
            correct += preds.eq(change_labels).sum().item()
            
            labels_cpu = change_labels.cpu().numpy().tolist()

            visualize_activation_map(activation, layer_names, iter_, 'test', img_dir, preds_cpu, labels_cpu, base, fu)

        print('[*] Test Acc: {:5f}'.format(100.*correct/total))
        writer.add_scalar('Test acc', 100.*correct/total, iter_)

    model.train()


def main(args):
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True

    # 0. device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    print('[*] device: ', device)

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M%S'))
    folder_name = '{}_{}_{}'.format(today, args.message, args.dataset)
    
    log_dir = os.path.join(args.log_dir, folder_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, folder_name)

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # for log
    f = open(os.path.join(log_dir,'arguments.txt'), 'w')
    f.write(str(args))
    f.close()

    # make datasets & dataloader (train & test)
    print('[*] prepare datasets & dataloader...')

    train_datasets = ClassPairDataset(args.train_path, dataset=args.dataset, mode='train')
    test_datasets = ClassPairDataset(args.test_path, dataset=args.dataset, mode='test')
    
    print('[*] train data property')
    train_datasets.get_data_property()
    print('[*] test data property')
    test_datasets.get_data_property()
    
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, num_workers=args.w, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=6, num_workers=4, pin_memory=True, shuffle=True)

    # select network
    print('[*] build network...')
    model = acm_resnet152(num_classes=512)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.8, milestones=[3, 5, 7]) 
    
    if args.resume is True:
        checkpoint = torch.load(args.pretrained)
        args.start_epoch = checkpoint['epoch']
        args.start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    model = model.to(device)

    # training
    print('[*] start training...')
    summary_writer = SummaryWriter(log_dir)
    train(args, train_loader, test_loader, model, optimizer, scheduler, device, summary_writer, log_dir, checkpoint_dir)


if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
