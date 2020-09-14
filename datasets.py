import os
import sys
import numpy as np
import torch
import random
import glob

from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

from config import parse_arguments
from PIL import Image

import json

class ClassPairDataset(Dataset):
    def __init__(self, input_path, dataset, mode, transform=None):
        if dataset == 'toy':
            input_path = input_path + '_toy'
        
        self.input_path = os.path.join(input_path, '{}set/512'.format(mode))
        self.disease_label_path = os.path.join(input_path, 'label_csv/disease.json')
        with open(self.disease_label_path, "r") as f:
            self.disease_label = json.load(open(self.disease_label_path))
        
        json_name = '4class_datasets_{}_512_{}.json'.format(dataset,mode)
        if os.path.exists(json_name) is True:
            print('[*] {} is already exist. Loading Json from {}'.format(json_name, json_name))
            with open(json_name, "r") as f:
                self.samples = json.load(open(json_name,"r"))
        else:
            print('[*] There is no {}. Start making new Json'.format(json_name, json_name))
            self.samples = self._make_dataset(mode)
            with open(json_name, "w") as f:
                json.dump(self.samples, f)

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(580),
                transforms.CenterCrop(512),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=(0.7, 1.3),
                        contrast=(0.7, 1.3),
                        saturation=(0.7, 1.3),
                        hue=(-0.1, 0.1))
                    ], p=0.5),
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.RandomErasing(scale=(0.02,0.1))
                    ], p=0.2),
                ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor(),
                ])
    
    def _find_disease_label(self, exam_id):
        if exam_id in self.disease_label['normal']:
            return 0 #normal
        elif exam_id in self.disease_label['abnormal']:
            return 1 #abnormal
        else:
            return 2

    def _make_dataset(self, phase):
        if phase == 'train':
            categories = os.listdir(self.input_path)

            samples = {'imgs':[], 'change_labels':[], 'disease_labels':[]}
            
            other_label_cnt = 0
            missing_pair_cnt = 0

            for category in categories:
                change_label = 1 if category.split('_')[-1]=='nochange' else 0
                
                category_path = os.path.join(self.input_path, category)
                for root, dirs, files in os.walk(category_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        research_id_path = '/'.join(file_path.split('/')[:-1])

                        idx_bf_flag = file.split('_')[0]
                        bf_flag = idx_bf_flag[-1]
                        idx_flag = idx_bf_flag[:-1]

                        if bf_flag == 'b':
                            pair_flag = 'f'
                            pair_path = glob.glob(research_id_path + '/' + idx_flag + pair_flag + '_' + '*')

                            if len(pair_path) > 0:
                                pair_path = pair_path[0]

                                bf_exam = pair_path.split('/')[-1]
                                exam_id = bf_exam.split('_')[-1]
                                exam_id = exam_id.split('.')[0]
                                
                                pairs =[file_path, pair_path]

                                disease_label = []
                                for pair in pairs:
                                    search_id = pair.split('/')[-1].split('.')[0]
                                    _label = self._find_disease_label(search_id)
                                    if _label > 1:
                                        other_label_cnt += 1
                                        break
                                    else:
                                        disease_label.append(_label)

                                if len(disease_label) > 0:
                                    samples['imgs'].append(pairs)
                                    samples['change_labels'].append(change_label)
                                    samples['disease_labels'].append(disease_label)
                            else:
                                missing_pair_cnt += 1 
        else:
            categories = os.listdir(self.input_path)

            samples = {'imgs':[], 'change_labels':[], 'disease_labels':[]}
            
            other_label_cnt = 0
            missing_pair_cnt = 0

            for category in categories:
                change_label = 1 if category.split('_')[-1]=='nochange' else 0
                
                category_path = os.path.join(self.input_path, category)
                for root, dirs, files in os.walk(category_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        research_id_path = '/'.join(file_path.split('/')[:-1])

                        idx_bf_flag = file.split('_')[0]
                        bf_flag = idx_bf_flag[-1]
                        idx_flag = idx_bf_flag[:-1]

                        if bf_flag == 'b':
                            pair_flag = 'f'
                            pair_path = glob.glob(research_id_path + '/' + idx_flag + pair_flag + '_' + '*')

                            if len(pair_path) > 0:
                                pair_path = pair_path[0]

                                bf_exam = pair_path.split('/')[-1]
                                exam_id = bf_exam.split('_')[-1]
                                exam_id = exam_id.split('.')[0]
                                
                                pairs =[file_path, pair_path]

                                disease_label = []
                                for pair in pairs:
                                    search_id = pair.split('/')[-1].split('.')[0]
                                    _label = self._find_disease_label(search_id)
                                    if _label > 1:
                                        other_label_cnt += 1
                                        break
                                    else:
                                        disease_label.append(_label)

                                if len(disease_label) > 0:
                                    samples['imgs'].append(pairs)
                                    samples['change_labels'].append(change_label)
                                    samples['disease_labels'].append([0,0])
                            else:
                                missing_pair_cnt += 1 


        return samples

    def __getitem__(self, idx):
        base_img = self.transform(Image.open(self.samples['imgs'][idx][0]))
        base_img = self._catch_exception(base_img)
        pair_img = self.transform(Image.open(self.samples['imgs'][idx][1]))
        pair_img = self._catch_exception(pair_img)

        change_labels = self.samples['change_labels'][idx]
        disease_labels = self.samples['disease_labels'][idx]

        return base_img, pair_img, change_labels, disease_labels
            
    def __len__(self):
        return len(self.samples['change_labels'])
    
    def _catch_exception(self, img):
        return img[0, :, :].unsqueeze(0) if img.shape[0] == 3 else img
        

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
