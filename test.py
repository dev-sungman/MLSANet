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
import cv2

def register_forward_hook(model):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    layer_names = ['vis1', 'vis2']
    #layer_names = ['layer2', 'layer3']
    model.vis_final1.register_forward_hook(get_activation(layer_names[0]))
    model.vis_final2.register_forward_hook(get_activation(layer_names[1]))
    
    return activation, layer_names

def visualize_activation_map(activation, layer_names, iter_, phase, img_dir, preds, labels, base, fu, model, likeli):
    label_names = ['change', 'nochange']
    acts = []
    num_layers = len(layer_names)
    normalize = nn.Softmax(dim=2)

    visual_num = 6
    
    for layer in layer_names:
        act = activation[layer].squeeze()
        if len(act.shape) > 3:
            b, c, h, w = act.shape
        
            act = torch.mean(act, dim=1)
            act -= act.min(1, keepdim=True)[0]
            act /= act.max(1, keepdim=True)[0]
            acts.append(act)

    if len(acts) > 0:
        for batch in range(visual_num): #batch
            np_base = base[batch,0,:,:].cpu().detach().numpy()
            np_base = cv2.resize(np_base, (256, 256))
            np_base = cv2.cvtColor(np_base, cv2.COLOR_GRAY2BGR)
            np_fu = fu[batch,0,:,:].cpu().detach().numpy()
            np_fu = cv2.resize(np_base, (256, 256))
            np_fu = cv2.cvtColor(np_fu, cv2.COLOR_GRAY2BGR)

            np_base_act = acts[0][batch,:,:].cpu().detach().numpy()
            np_fu_act = acts[1][batch,:,:].cpu().detach().numpy()

            print(np.min(np_base_act), np.max(np_base_act))

            np_base_act = cv2.resize(np_base_act, (256,256))
            np_base_act -= 0.8
            np_base_act[np_base_act<0] = 0.
            np_base_act -= np.min(np_base_act)
            np_base_act /= np.max(np_base_act)
            np_fu_act = cv2.resize(np_fu_act, (256,256))
            np_fu_act -= 0.8
            np_fu_act[np_fu_act<0] = 0.
            np_fu_act -= np.min(np_fu_act)
            np_fu_act /= np.max(np_fu_act)

            base_heat = cv2.applyColorMap(np.uint8(255*np_base_act), cv2.COLORMAP_BONE)
            base_heat = np.float32(base_heat) /255
            fu_heat = cv2.applyColorMap(np.uint8(255*np_fu_act), cv2.COLORMAP_BONE)
            fu_heat = np.float32(fu_heat) /255

            base_cam = np.float32(np_base) * base_heat
            base_cam = base_cam / np.max(base_cam)
            
            fu_cam = np.float32(np_fu) * fu_heat
            fu_cam = fu_cam / np.max(fu_cam)

            
            label_name = label_names[labels[batch]]
            pred_name = label_names[preds[batch]]

            if (label_name == 'nochange') & (pred_name == 'nochange'):
                TP_path = os.path.join(img_dir, 'tp')
                pathlib.Path(TP_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(TP_path,'{}_base.jpg'.format(iter_)), np_base*255)
                cv2.imwrite(os.path.join(TP_path,'{}_fu.jpg'.format(iter_)), np_fu*255)
                cv2.imwrite(os.path.join(TP_path,'{}_base_am.jpg'.format(iter_)), base_cam*255)
                cv2.imwrite(os.path.join(TP_path,'{}_fu_am.jpg'.format(iter_)), fu_cam*255)
            elif (label_name == 'nochange') & (pred_name == 'change'):
                FN_path = os.path.join(img_dir, 'fn')
                pathlib.Path(FN_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(FN_path,'{}_base.jpg'.format(iter_)), np_base*255)
                cv2.imwrite(os.path.join(FN_path,'{}_fu.jpg'.format(iter_)), np_fu*255)
                cv2.imwrite(os.path.join(FN_path,'{}_base_am.jpg'.format(iter_)), base_cam*255)
                cv2.imwrite(os.path.join(FN_path,'{}_fu_am.jpg'.format(iter_)), fu_cam*255)
            elif (label_name == 'change') & (pred_name == 'change'):
                TN_path = os.path.join(img_dir, 'tn')
                pathlib.Path(TN_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(TN_path,'{}_base.jpg'.format(iter_)), np_base*255)
                cv2.imwrite(os.path.join(TN_path,'{}_fu.jpg'.format(iter_)), np_fu*255)
                cv2.imwrite(os.path.join(TN_path,'{}_base_am.jpg'.format(iter_)), base_cam*255)
                cv2.imwrite(os.path.join(TN_path,'{}_fu_am.jpg'.format(iter_)), fu_cam*255)
            else:
                FP_path = os.path.join(img_dir, 'fp')
                pathlib.Path(FP_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(FP_path,'{}_base.jpg'.format(iter_)), np_base*255)
                cv2.imwrite(os.path.join(FP_path,'{}_fu.jpg'.format(iter_)), np_fu*255)
                cv2.imwrite(os.path.join(FP_path,'{}_base_am.jpg'.format(iter_)), base_cam*255)
                cv2.imwrite(os.path.join(FP_path,'{}_fu_am.jpg'.format(iter_)), fu_cam*255)

        

def test(args, data_loader, model, device, log_dir, checkpoint_dir):
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
        
        preds_cpu = preds.cpu().numpy().tolist()
        labels_cpu = labels.cpu().numpy().tolist()
        overall_pred += preds_cpu
        overall_gt += labels_cpu

        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        visualize_activation_map(activation, layer_names, iter_, 'test', img_dir, preds_cpu, labels_cpu, base, fu, model, preds_cpu)
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
