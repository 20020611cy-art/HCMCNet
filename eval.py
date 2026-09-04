import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.changeDataset import ChangeDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from PIL import Image
import cv2
import numpy as np

#---------------后处理----------------
def post_process(mask, min_area=120):
    """
    mask: numpy array, 0/1 二值图
    min_area: 连通域面积过滤阈值
    """
    mask = mask.astype(np.uint8)

    # --- 1) 闭运算：填洞、补边 ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 开运算：去毛刺、小噪声
   #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # --- 2) 连通域去除小区域 ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            mask[labels == i] = 0


    return mask
#---------------------------------------

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, config):
        As = data['A']
        Bs = data['B']
        label = data['gt']
        name = data['fn']
        pred = self.sliding_eval_rgbX(As, Bs, config.eval_crop_size, config.eval_stride_rate, device)

        # pred = post_process(pred, min_area=60)#加入后处理

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)

            fn = name + '.png'

            # 保存二值图：背景黑色0，变化白色255
            pred_binary = (pred > 0).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(self.save_path, fn), pred_binary)
            logger.info('Save the image ' + fn)

        if self.show_image:
            # 显示二值图
            pred_binary = (pred > 0).astype(np.uint8) * 255
            cv2.imshow('pred_binary', pred_binary)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, recall, precision, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc,f1,kappa = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc,
                               self.dataset.class_names, show_no_back=False)
        print("IoU:", iou)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("OA:", pixel_acc)
        print("Kappa:", kappa)
        return result_line,mean_IoU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--dataset_name', '-n', default='mfnet', type=str)
    parser.add_argument('--split', '-c', default='test', type=str)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    
    dataset_name = args.dataset_name
    if dataset_name == 'levir':
        from configs.config_levir import config
    elif dataset_name == 'dsifn':
        from configs.config_dsifn import config
    elif dataset_name == 'whu':
        from configs.config_whu import config
    elif dataset_name == 'levirjia':
        from configs.config_levirjia import config
    elif dataset_name == 'sysu':
        from configs.config_sysu import config
    else:
        raise ValueError('Not a valid dataset name')

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    flops = network.flops()
    print("Gflops of the network: ", flops/(10**9))
    print("number of paramters: ", sum(p.numel() if p.requires_grad==True else 0 for p in network.parameters()))
    #1/0
    data_setting = {'root': config.root_folder,
                    'A_format': config.A_format,
                    'B_format': config.B_format,
                    'gt_format': config.gt_format,
                    'class_names': config.class_names}
    val_pre = ValPre()
    dataset = ChangeDataset(data_setting, args.split, val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image, config)
        _, mean_IoU = segmentor.run_eval(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)

    #visualize erf

    # with torch.enable_grad():
    #     segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, network,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, args.verbose, args.save_path,
    #                                 args.show_image, config)
        
    #     segmentor.get_erf(config.checkpoint_dir, args.epochs)