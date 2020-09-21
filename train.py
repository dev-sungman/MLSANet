import os
import sys

import numpy as np
from config import parse_arguments
from datasets import ClassPairDataset
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

def ACM_loss(logit):
    return 2-(2*logit)

def register_forward_hook(model):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    layer_names = ['k_conv1', 'q_conv1', 'k_conv2', 'q_conv2', 'k_conv3', 'q_conv3', 'k_conv4', 'q_conv4']
    model.acm1.k_conv.register_forward_hook(get_activation(layer_names[0]))
    model.acm1.q_conv.register_forward_hook(get_activation(layer_names[1]))
    model.acm2.k_conv.register_forward_hook(get_activation(layer_names[2]))
    model.acm2.q_conv.register_forward_hook(get_activation(layer_names[3]))
    model.acm3.k_conv.register_forward_hook(get_activation(layer_names[4]))
    model.acm3.q_conv.register_forward_hook(get_activation(layer_names[5]))
    model.acm4.k_conv.register_forward_hook(get_activation(layer_names[6]))
    model.acm4.q_conv.register_forward_hook(get_activation(layer_names[7]))
    
    return activation, layer_names

'''
def register_forward_hook(model):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    model.layer1.register_forward_hook(get_activation(layer_names[0]))
    model.layer2.register_forward_hook(get_activation(layer_names[1]))
    model.layer3.register_forward_hook(get_activation(layer_names[2]))
    model.layer4.register_forward_hook(get_activation(layer_names[3]))
    
    return activation, layer_names
'''

def visualize_activation_map(activation, layer_names, iter_, phase, img_dir, thresh=0.85):
    acts = []
    num_layers = len(layer_names)
    normalize = nn.Softmax(dim=2)

    for layer in layer_names:
        act = activation[layer].squeeze()
        
        b, c, h, w = act.shape
        
        act = torch.mean(act, dim=1)
        act = act.view(b, 1, h*w)
        act = normalize(act)
        act = act.view(b, h, w)

        acts.append(act)

    fig, axarr = plt.subplots(len(layer_names), acts[0].shape[0], figsize=(60,40))
    
    for ia, act in enumerate(acts): # 6
        for batch in range(5): #batch
            axarr[ia, batch].imshow(act[batch,:,:].cpu().detach().numpy())

            axarr[ia, batch].set_title('{}_{}'.format(layer_names[ia],batch))
    
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)                
    plt.savefig(os.path.join(img_dir, 'act_map_{}_{}.png').format(phase, iter_))
    plt.close(fig)
    plt.clf()

def train(args, data_loader, test_loader, model, device, writer, log_dir, checkpoint_dir):
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=args.lr) 
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=0.1)
    correct = 0
    total = 0
    
    overall_iter = 0
    img_dir = os.path.join(log_dir, 'imgs')
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # NEED TO UPDATE DATSETS. 
        iter_ = 0 
        running_loss = 0
        running_orth = 0
        running_change = 0
        running_disease = 0

        for base, fu, change_labels, disease_labels in iter(data_loader):
            base = base.to(device)
            fu = fu.to(device)
            change_labels = change_labels.to(device)
            disease_labels = [disease_labels[0].to(device), disease_labels[1].to(device)]

            # for activation
            activation, layer_names = register_forward_hook(model)

            base_embed, fu_embed, outputs, orth = model(base,fu)
            
            _, preds = outputs.max(1)
            total += change_labels.size(0)
            correct += preds.eq(change_labels).sum().item()
            
            # change loss
            ce_criterion = nn.CrossEntropyLoss()
            change_loss = ce_criterion(outputs, change_labels)

            # orthogonal loss
            orth_loss = ACM_loss(orth)

            # disease loss
            disease_loss = ce_criterion(base_embed, disease_labels[0]) + ce_criterion(fu_embed, disease_labels[1])


            #overall_loss = change_loss + (orth_loss*0.4) + (disease_loss*0.8)
            #overall_loss = change_loss + disease_loss
            overall_loss = change_loss + disease_loss + orth_loss
            
            running_change += change_loss.item()
            running_orth += orth_loss.item()
            running_disease += disease_loss.item()
            running_loss += overall_loss.item()

            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()
            
            if (iter_ % args.print_freq == 0) & (iter_ != 0):
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                print('Epoch: {:2d}, LR: {:5f}, Iter: {:5d}, Cls loss: {:5f}, Orth loss: {:5f}, Disease loss: {:5f}, Overall loss: {:5f}, Acc: {:4f}'.format(epoch, lr, iter_, running_change/iter_, running_orth/iter_, running_loss/iter_, running_disease/iter_, 100.*correct/total))
                writer.add_scalar('change_loss', running_change/iter_, overall_iter)
                writer.add_scalar('orth_loss', running_orth/iter_, overall_iter)
                writer.add_scalar('disease_loss', running_disease/iter_, overall_iter)
                
                visualize_activation_map(activation, layer_names, overall_iter, 'train', img_dir)
                
            iter_ += 1
            overall_iter += 1
        
        scheduler.step()
        test(args, test_loader, model, device, writer, log_dir, checkpoint_dir, overall_iter)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, str(overall_iter)) + '.pth')


def test(args, data_loader, model, device, writer, log_dir, checkpoint_dir, iter_):
    model.eval()
    
    correct = 0
    total = 0
    img_dir = os.path.join(log_dir, 'imgs')
    
    with torch.no_grad():
        for base, fu, labels, _ in iter(data_loader):
            base = base.to(device)
            fu = fu.to(device)
            labels = labels.to(device)
            
            activation, layer_names = register_forward_hook(model)
            _, _, outputs, _ = model(base,fu)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            visualize_activation_map(activation, layer_names, iter_, 'test', img_dir)

        print('[*] Test Acc: {:5f}'.format(100.*correct/total))
        writer.add_scalar('test_acc', 100.*correct/total, iter_)

    model.train()

def main(args):
    # 0. device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

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

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, num_workers=args.w, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    # select network
    print('[*] build network...')
    net = acm_resnet50(num_classes=512)
    #net = acm_resnet152(num_classes=512)

    if torch.cuda.device_count() > 1 and device=='cuda':
        net = nn.DataParallel(net)
    
    net = net.to(device)

    # training
    print('[*] start training...')
    summary_writer = SummaryWriter(log_dir)
    train(args, train_loader, test_loader, net, device, summary_writer, log_dir, checkpoint_dir)


if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
