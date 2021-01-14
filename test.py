import sys
import os
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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import cv2

import json
import itertools

def save_results_metric(tn, tp, fn, fp, correct, total, log_dir)
    tp, fn, fp, tn = tp.item(), fn.item(), fp.item(), tn.item()
    results_dict = {}
    results_dict['tn'] = tn
    results_dict['tp'] = tp
    results_dict['fn'] = fn
    results_dict['fp'] = fp
    results_dict['specificity'] = tn/(tn+fp)
    results_dict['sensitivity'] = tp/(tp+fn)
    results_dict['ppv'] = tp/(tp+fp)
    results_dict['npv'] = tn/(tn+fn)
    results_dict['acc'] = 100.*correct/total

    print('tn, fp, fn, tp: ', tn, fp, fn, tp)
    print('specificity: ', tn/(tn+fp))
    print('sensitivity: ', tp/(tp+fn))
    print('positive predictive value: ', tp/(tp+fp))
    print('negative predictive value: ', tn/(tn+fn))
    print('test_acc: ', 100.*correct/total)
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dumps(results_dict, f)

def save_roc_auc_curve(overall_gt, overall_output)
    ### ROC, AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    np_gt = np.array(overall_gt)
    np_output = np.array(overall_output)
    fpr, tpr, _ = roc_curve(np_gt, np_output, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(log_dir, 'roc_auc.png'))

def save_confusion_matrix(cm, target_names, log_dir, title='CFMatrix', cmap=None, normalize=True):
    acc = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - acc

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i,j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i,j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n accuracy={:0.4f}'.format(acc))
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))

def test(args, data_loader, model, device, log_dir):
    print('[*] Test Phase')
    model.eval()
    
    correct = 0
    total = 0
    
    img_dir = os.path.join(log_dir, 'imgs')
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)
    
    overall_output = []
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
        outputs = F.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        new_labels = []
        for i in range(labels.shape[0]):
           new_labels.append(outputs[i, 1].cpu().detach().item())
        
        preds_cpu = preds.cpu().detach().numpy().tolist()
        labels_cpu = labels.cpu().detach().numpy().tolist()
        overall_output += new_labels
        overall_pred += preds_cpu
        overall_gt += labels_cpu

        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        visualize_activation_map(activation, layer_names, iter_, 'test', img_dir, preds_cpu, labels_cpu, base, fu)
        iter_ += 1

    tn, fp, fn, tp = confusion_matrix(overall_gt, overall_pred).ravel()
    save_results_metric(tn, tp, fn, fp, correct, total, log_dir)
    save_confusion_matrix(confusion_matrix(overall_gt, overall_pred), ['Change','No-Change'], log_dir)
    save_roc_auc_curve(overall_gt, overall_output)
        

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
