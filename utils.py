import cv2
import numpy as np
import pathlib
import torch
import os

def register_forward_hook(model):
    activation = {}
    grads = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    def get_grads(name):
        def hook(model, _in, _out):
            grads[name] = _out[0].detach()
        return hook

    layer_names = ['vis1', 'vis2']
    model.vis_final1.register_forward_hook(get_activation(layer_names[0]))
    model.vis_final2.register_forward_hook(get_activation(layer_names[1]))

    return activation, layer_names

def visualize_activation_map(activation, layer_names, iter_, phase, img_dir, preds, labels, base, fu):
    img_mean = 0.2
    img_std = 0.4
    label_names = ['change', 'nochange']
    acts = []
    num_layers = len(layer_names)
    
    visual_num = 6
    if base.shape[0] < visual_num:
        visual_num = base.shape[0]
    
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
            np_base = cv2.resize(np_base, (512, 512))
            np_base = cv2.cvtColor(np_base, cv2.COLOR_GRAY2BGR)
            np_base = (np_base*img_std)+img_mean
            
            np_fu = fu[batch,0,:,:].cpu().detach().numpy()
            np_fu = cv2.resize(np_fu, (512, 512))
            np_fu = cv2.cvtColor(np_fu, cv2.COLOR_GRAY2BGR)
            np_fu = (np_fu*img_std) + img_mean

            np_base_act = acts[0][batch,:,:].cpu().detach().numpy()
            np_fu_act = acts[1][batch,:,:].cpu().detach().numpy()

            np_base_act = cv2.resize(np_base_act, (512,512))
            np_base_act -= 0.8
            np_base_act[np_base_act<0] = 0.
            np_base_act -= np.min(np_base_act)
            np_base_act /= np.max(np_base_act)

            np_fu_act = cv2.resize(np_fu_act, (512,512))
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
            
            full_image = np.zeros((1024,1024,3))
            full_image[:512, :512, :] = np_base*255
            full_image[512:, :512, :] = np_fu*255
            full_image[:512, 512:, :] = base_cam*255
            full_image[512:, 512:, :] = fu_cam*255

            if (label_name == 'nochange') & (pred_name == 'nochange'):
                TP_path = os.path.join(img_dir, 'tp')
                pathlib.Path(TP_path).mkdir(parents=True, exist_ok=True)
                #cv2.imwrite(os.path.join(TP_path,'{}_{}_base_{}.jpg'.format(iter_, phase, batch)), np_base*255)
                #cv2.imwrite(os.path.join(TP_path,'{}_{}_fu_{}.jpg'.format(iter_, phase, batch)), np_fu*255)
                #cv2.imwrite(os.path.join(TP_path,'{}_{}_base_am_{}.jpg'.format(iter_, phase, batch)), base_cam*255)
                #cv2.imwrite(os.path.join(TP_path,'{}_{}_fu_am_{}.jpg'.format(iter_, phase, batch)), fu_cam*255)
                cv2.imwrite(os.path.join(TP_path,'{}_{}_full_{}.jpg'.format(iter_, phase, batch)), full_image)
            elif (label_name == 'nochange') & (pred_name == 'change'):
                FN_path = os.path.join(img_dir, 'fn')
                pathlib.Path(FN_path).mkdir(parents=True, exist_ok=True)
                #cv2.imwrite(os.path.join(FN_path,'{}_{}_base_{}.jpg'.format(iter_, phase, batch)), np_base*255)
                #cv2.imwrite(os.path.join(FN_path,'{}_{}_fu_{}.jpg'.format(iter_, phase, batch)), np_fu*255)
                #cv2.imwrite(os.path.join(FN_path,'{}_{}_base_am_{}.jpg'.format(iter_, phase, batch)), base_cam*255)
                #cv2.imwrite(os.path.join(FN_path,'{}_{}_fu_am_{}.jpg'.format(iter_, phase, batch)), fu_cam*255)
                cv2.imwrite(os.path.join(FN_path,'{}_{}_full_{}.jpg'.format(iter_, phase, batch)), full_image)
            elif (label_name == 'change') & (pred_name == 'change'):
                TN_path = os.path.join(img_dir, 'tn')
                pathlib.Path(TN_path).mkdir(parents=True, exist_ok=True)
                #cv2.imwrite(os.path.join(TN_path,'{}_{}_base_{}.jpg'.format(iter_, phase, batch)), np_base*255)
                #cv2.imwrite(os.path.join(TN_path,'{}_{}_fu_{}.jpg'.format(iter_, phase, batch)), np_fu*255)
                #cv2.imwrite(os.path.join(TN_path,'{}_{}_base_am_{}.jpg'.format(iter_, phase, batch)), base_cam*255)
                #cv2.imwrite(os.path.join(TN_path,'{}_{}_fu_am_{}.jpg'.format(iter_, phase, batch)), fu_cam*255)
                cv2.imwrite(os.path.join(TN_path,'{}_{}_full_{}.jpg'.format(iter_, phase, batch)), full_image)
            else:
                FP_path = os.path.join(img_dir, 'fp')
                pathlib.Path(FP_path).mkdir(parents=True, exist_ok=True)
                #cv2.imwrite(os.path.join(FP_path,'{}_{}_base_{}.jpg'.format(iter_, phase, batch)), np_base*255)
                #cv2.imwrite(os.path.join(FP_path,'{}_{}_fu_{}.jpg'.format(iter_, phase, batch)), np_fu*255)
                #cv2.imwrite(os.path.join(FP_path,'{}_{}_base_am_{}.jpg'.format(iter_, phase, batch)), base_cam*255)
                #cv2.imwrite(os.path.join(FP_path,'{}_{}_fu_am_{}.jpg'.format(iter_, phase, batch)), fu_cam*255)
                cv2.imwrite(os.path.join(FP_path,'{}_{}_full_{}.jpg'.format(iter_, phase, batch)), full_image)
