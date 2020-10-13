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

import time
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def register_forward_hook(model):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    layer_names = ['vis1', 'vis2']
    model.vis_final1.register_forward_hook(get_activation(layer_names[0]))
    model.vis_final2.register_forward_hook(get_activation(layer_names[1]))
    
    return activation, layer_names

def visualize_activation_map(activation, layer_names, iter_, phase, img_dir, thresh=0.85):
    acts = []
    num_layers = len(layer_names)
    normalize = nn.Softmax(dim=2)

    visual_num = 6

    for layer in layer_names:
        act = activation[layer].squeeze()
        print(act.shape)
        if len(act.shape) > 3:
            b, c, h, w = act.shape
        
            act = torch.mean(act, dim=1)
            act = act.view(b, 1, h*w)
            act = normalize(act)
            act = act.view(b, h, w)

            acts.append(act)

    fig, axarr = plt.subplots(len(layer_names), visual_num, figsize=(15,10))
    
    for ia, act in enumerate(acts): # 6
        for batch in range(visual_num): #batch
            axarr[ia, batch].imshow(act[batch,:,:].cpu().detach().numpy())

            axarr[ia, batch].set_title('{}_{}'.format(layer_names[ia],batch))
    
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0)                
    plt.savefig(os.path.join(img_dir, 'act_map_{}_{}.png').format(phase, iter_))
    plt.close(fig)
    plt.clf()

def test(args, data_loader, model, device, log_dir, checkpoint_dir):
    print('[*] Test Phase')
    model.eval()
    
    correct = 0
    total = 0
    
    img_dir = os.path.join(log_dir, 'imgs')
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        overall_pred = []
        overall_gt = []
        iter_ = 0
        for base, fu, labels, _ in iter(data_loader):
            base = base.to(device)
            fu = fu.to(device)
            labels = labels.to(device)
            
            # for activation
            activation, layer_names = register_forward_hook(model)
            
            _, _, outputs, _ = model(base,fu)
            _, preds = outputs.max(1)
            
            preds_cpu = preds.cpu().numpy().tolist()
            labels_cpu = labels.cpu().numpy().tolist()
            overall_pred += preds_cpu
            overall_gt += labels_cpu

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            
            visualize_activation_map(activation, layer_names, iter_, 'test', img_dir)
            iter_ += 1

        print(overall_pred, overall_gt) 
        tn, fp, fn, tp = confusion_matrix(overall_gt, overall_pred).ravel()
        print('tn, fp, fn, tp: ', tn, fp, fn, tp)
        print('specificity: ', tn/(tn+fp))
        print('sensitivity: ', tp/(tp+fn))
        print('positive predictive value: ', tp/(tp+fp))
        print('negative predictive value: ', tn/(tn+fn))
        print('test_acc: ', 100.*correct/total)

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

    test_datasets = ClassPairDataset(args.test_path, dataset=args.dataset, mode='test')

    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    # select network
    print('[*] build network...')
    #net = acm_resnet50(num_classes=512)
    net = acm_resnet152(num_classes=512)
    net.load_state_dict(torch.load(args.pretrained))

    if torch.cuda.device_count() > 1 and device=='cuda':
        net = nn.DataParallel(net)
    
    net = net.to(device)

    test(args, test_loader, net, device, log_dir, checkpoint_dir)

if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
