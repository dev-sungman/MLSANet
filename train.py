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
import cv2

def ACM_loss(logit):
    return 2-(2*logit)

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

def visualize_activation_map(activation, layer_names, iter_, phase, img_dir, preds, labels, base, fu):
    label_names = ['change', 'nochange']
    acts = []
    num_layers = len(layer_names)

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
            np_fu = cv2.resize(np_fu, (256, 256))
            np_fu = cv2.cvtColor(np_fu, cv2.COLOR_GRAY2BGR)

            np_base_act = acts[0][batch,:,:].cpu().detach().numpy()
            np_fu_act = acts[1][batch,:,:].cpu().detach().numpy()

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


            base_heat = cv2.applyColorMap(np.uint8(255*np_base_act), cv2.COLORMAP_JET)
            base_heat = np.float32(base_heat) /255
            fu_heat = cv2.applyColorMap(np.uint8(255*np_fu_act), cv2.COLORMAP_JET)
            fu_heat = np.float32(fu_heat) /255

            base_cam = np.float32(np_base) + base_heat
            base_cam = base_cam / np.max(base_cam)
            
            fu_cam = np.float32(np_fu) + fu_heat
            fu_cam = fu_cam / np.max(fu_cam)
            
            label_name = label_names[labels[batch]]
            pred_name = label_names[preds[batch]]
            if (label_name == 'nochange') & (pred_name == 'nochange'):
                TP_path = os.path.join(img_dir, 'tp')
                pathlib.Path(TP_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(TP_path,'{}_{}_base.jpg'.format(iter_, phase)), np_base*255)
                cv2.imwrite(os.path.join(TP_path,'{}_{}_fu.jpg'.format(iter_, phase)), np_fu*255)
                cv2.imwrite(os.path.join(TP_path,'{}_{}_base_am.jpg'.format(iter_, phase)), base_cam*255)
                cv2.imwrite(os.path.join(TP_path,'{}_{}_fu_am.jpg'.format(iter_, phase)), fu_cam*255)
            elif (label_name == 'nochange') & (pred_name == 'change'):
                FN_path = os.path.join(img_dir, 'fn')
                pathlib.Path(FN_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(FN_path,'{}_{}_base.jpg'.format(iter_, phase)), np_base*255)
                cv2.imwrite(os.path.join(FN_path,'{}_{}_fu.jpg'.format(iter_, phase)), np_fu*255)
                cv2.imwrite(os.path.join(FN_path,'{}_{}_base_am.jpg'.format(iter_, phase)), base_cam*255)
                cv2.imwrite(os.path.join(FN_path,'{}_{}_fu_am.jpg'.format(iter_, phase)), fu_cam*255)
            elif (label_name == 'change') & (pred_name == 'change'):
                TN_path = os.path.join(img_dir, 'tn')
                pathlib.Path(TN_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(TN_path,'{}_{}_base.jpg'.format(iter_, phase)), np_base*255)
                cv2.imwrite(os.path.join(TN_path,'{}_{}_fu.jpg'.format(iter_, phase)), np_fu*255)
                cv2.imwrite(os.path.join(TN_path,'{}_{}_base_am.jpg'.format(iter_, phase)), base_cam*255)
                cv2.imwrite(os.path.join(TN_path,'{}_{}_fu_am.jpg'.format(iter_, phase)), fu_cam*255)
            else:
                FP_path = os.path.join(img_dir, 'fp')
                pathlib.Path(FP_path).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(FP_path,'{}_{}_base.jpg'.format(iter_, phase)), np_base*255)
                cv2.imwrite(os.path.join(FP_path,'{}_{}_fu.jpg'.format(iter_, phase)), np_fu*255)
                cv2.imwrite(os.path.join(FP_path,'{}_{}_base_am.jpg'.format(iter_, phase)), base_cam*255)
                cv2.imwrite(os.path.join(FP_path,'{}_{}_fu_am.jpg'.format(iter_, phase)), fu_cam*255)

def train(args, data_loader, test_loader, model, device, writer, log_dir, checkpoint_dir):
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 30]) 
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
            
            preds_cpu = preds.cpu().numpy().tolist()
            labels_cpu = change_labels.cpu().numpy().tolist()
            
            # change loss
            ce_criterion = nn.CrossEntropyLoss()
            change_loss = ce_criterion(outputs, change_labels)

            # orthogonal loss
            orth_loss = ACM_loss(orth)

            # disease loss
            disease_loss = ce_criterion(base_embed, disease_labels[0]) + ce_criterion(fu_embed, disease_labels[1])

            #overall_loss = change_loss + disease_loss
            #overall_loss = change_loss + disease_loss + (0.5*orth_loss)
            overall_loss = disease_loss
            
            running_change += change_loss.item()
            running_orth += orth_loss.item()
            running_disease += disease_loss.item()
            running_loss += overall_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
                writer.add_scalar('train_acc', 100.*correct/total, overall_iter)
                visualize_activation_map(activation, layer_names, overall_iter, 'train', img_dir, preds_cpu, labels_cpu, base, fu)
                
            iter_ += 1
            overall_iter += 1
        
        scheduler.step()
        test(args, test_loader, model, device, writer, log_dir, checkpoint_dir, overall_iter)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, str(overall_iter)) + '.pth')


def test(args, data_loader, model, device, writer, log_dir, checkpoint_dir, iter_):
    print('[*] Test Phase')
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

            preds_cpu = preds.cpu().numpy().tolist()
            labels_cpu = labels.cpu().numpy().tolist()

            visualize_activation_map(activation, layer_names, iter_, 'test', img_dir, preds_cpu, labels_cpu, base, fu)

        print('[*] Test Acc: {:5f}'.format(100.*correct/total))
        writer.add_scalar('test_acc', 100.*correct/total, iter_)

    model.train()

def main(args):
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

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, num_workers=args.w, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=6, num_workers=4, pin_memory=True, shuffle=True)

    # select network
    print('[*] build network...')
    #net = acm_resnet50(num_classes=512)
    net = acm_resnet152(num_classes=512)
    
    if args.resume is True:
        net.load_state_dict(torch.load(args.pretrained))

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
