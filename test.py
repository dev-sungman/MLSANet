import os
import sys

import numpy as np
from config import parse_arguments
from datasets import ClassPairDataset
from models.acm_resnet import acm_resnet50, acm_resnet152
from utils import register_forward_hook, visualize_activation_map

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import time
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2

def test(args, data_loader, model, device, log_dir):
    print('[*] Test Phase')
    model.eval()
    
    correct = 0
    total = 0
    
    img_dir = os.path.join(log_dir, 'imgs')
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)

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
        
        preds_cpu = preds.cpu().detach().numpy().tolist()
        labels_cpu = labels.cpu().detach().numpy().tolist()
        overall_pred += preds_cpu
        overall_gt += labels_cpu

        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        visualize_activation_map(activation, layer_names, iter_, 'test', img_dir, preds_cpu, labels_cpu, base, fu)
        iter_ += 1

    tp, fn, fp, tn = confusion_matrix(overall_gt, overall_pred).ravel()
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

    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M%S'))
    folder_name = '{}_{}_{}'.format(today, args.message, args.dataset)
    
    log_dir = os.path.join(args.log_dir, folder_name)

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # for log
    f = open(os.path.join(log_dir,'arguments.txt'), 'w')
    f.write(str(args))
    f.close()

    # make datasets & dataloader (train & test)
    print('[*] prepare datasets & dataloader...')

    test_datasets = ClassPairDataset(args.test_path, dataset=args.dataset, mode='test')

    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, num_workers=args.w, pin_memory=True, shuffle=True)

    # select network
    print('[*] build network...')
    model = acm_resnet152(num_classes=512)
    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint['state_dict'])
    
    model = model.cuda()

    test(args, test_loader, model, device, log_dir)

if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
