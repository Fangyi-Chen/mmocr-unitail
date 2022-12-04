'''generate ocr feature for hao for betr

'''
#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import cv2
import pickle
import mmcv
import numpy as np
import torch
from mmcv.image.misc import tensor2imgs
from mmcv.runner import load_checkpoint
from mmcv.utils.config import Config

from mmocr.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.utils.box_util import stitch_boxes_into_lines
from mmocr.utils.fileio import list_from_file
from mmocr.utils.model import revert_sync_batchnorm


# Parse CLI arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--img',
        type=str,
        help='Input image file or folder path.',
        default='ocr_feat/unitail-cls-aug')
    parser.add_argument(
        '--det_res',
        type=str,
        help='Input det folder path.',
        default='ocr_feat/det_res_aug/out_txt_dir')
    parser.add_argument(
        '--output',
        type=str,
        default='ocr_feat/ocrfeat_aug_2hao',
        help='Output file/folder name ocrfeat')
    parser.add_argument(
        '--recog',
        type=str,
        default='ABINet',
        help='Pretrained text recognition algorithm')
    parser.add_argument(
        '--recog-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected recog model. It'
        'overrides the settings in recog')
    parser.add_argument(
        '--recog-ckpt',
        type=str,
        default='./exps/fromweb/abinet_academic-f718abf6.pth',
        help='Path to the custom checkpoint file of the selected recog model. '
        'It overrides the settings in recog')
    parser.add_argument(
        '--kie',
        type=str,
        default='',
        help='Pretrained key information extraction algorithm')
    parser.add_argument(
        '--kie-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected kie model. It'
        'overrides the settings in kie')
    parser.add_argument(
        '--kie-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected kie model. '
        'It overrides the settings in kie')
    parser.add_argument(
        '--config-dir',
        type=str,
        default=os.path.join(str(Path.cwd()), 'configs/'),
        help='Path to the config directory where all the config files '
        'are located. Defaults to "configs/"')
    parser.add_argument(
        '--batch-mode',
        action='store_true',
        help='Whether use batch mode for inference')
    parser.add_argument(
        '--recog-batch-size',
        type=int,
        default=0,
        help='Batch size for text recognition')
    parser.add_argument(
        '--det-batch-size',
        type=int,
        default=0,
        help='Batch size for text detection')
    parser.add_argument(
        '--single-batch-size',
        type=int,
        default=0,
        help='Batch size for separate det/recog inference')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--export',
        type=str,
        default='',
        help='Folder where the results of each image are exported')
    parser.add_argument(
        '--export-format',
        type=str,
        default='json',
        help='Format of the exported result file(s)')
    parser.add_argument(
        '--details',
        action='store_true',
        help='Whether include the text boxes coordinates and confidence values'
    )
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Prints the recognised text')
    parser.add_argument(
        '--merge', action='store_true', help='Merge neighboring boxes')
    parser.add_argument(
        '--merge-xdist',
        type=float,
        default=20,
        help='The maximum x-axis distance to merge boxes')
    args = parser.parse_args()
    return args


class MMOCR:

    def __init__(self,
                 recog='SEG',
                 recog_config='',
                 recog_ckpt='',
                 config_dir=os.path.join(str(Path.cwd()), 'configs/'),
                 device='cuda:0',
                 **kwargs):

        textrecog_models = {
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'crnn/crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config': 'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt': 'sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'SAR_CN': {
                'config':
                'sar/sar_r31_parallel_decoder_chinese.py',
                'ckpt':
                'sar/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'satrn/satrn_academic_20211009-cb8b1580.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'satrn/satrn_small_20211009-2cf13355.pth'
            },
            'ABINet': {
                'config': 'abinet/abinet_academic.py',
                'ckpt': 'abinet/abinet_academic-f718abf6.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt': 'seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config': 'tps/crnn_tps_academic_dataset.py',
                'ckpt': 'tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
            }
        }

        self.tr = recog
        self.device = device

        # Check if the det/recog model choice is valid
        if self.tr and self.tr not in textrecog_models:
            raise ValueError(self.tr,
                             'is not a supported text recognition algorithm')
        self.recog_model = None
        if self.tr:
            # Build recognition model
            if not recog_config:
                recog_config = os.path.join(
                    config_dir, 'textrecog/',
                    textrecog_models[self.tr]['config'])
            if not recog_ckpt:
                recog_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'textrecog/' + textrecog_models[self.tr]['ckpt']

            self.recog_model = init_detector(
                recog_config, recog_ckpt, device=self.device)
            self.recog_model = revert_sync_batchnorm(self.recog_model)

        # Attribute check
        for model in list(filter(None, [self.recog_model])):
            if hasattr(model, 'module'):
                model = model.module

    def mkdir_savearray(self, path, array):
        folder = osp.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'wb') as f:
            pickle.dump(array, f, pickle.HIGHEST_PROTOCOL)
            # np.save(f, array)

    def read_dets_imgs(self, det_res, img, output):
        res_bank = []
        train_or_test = ['train', 'test', 'val']
        for t in train_or_test:
            classes = os.listdir(osp.join(det_res, t))
            for cls in classes:
                for txt in os.listdir(osp.join(det_res, t, cls)):
                    txt_path = osp.join(det_res, t, cls, txt)
                    img_path = osp.join(img, t, cls, txt[:-4]+'.jpg')
                    out_path = osp.join(output, t, cls, txt[:-4]+'.pkl')
                    f = open(txt_path, "r")
                    boxes =np.array([[float(b) for b in box.strip().split(',')] for box in f.readlines()])
                    f.close()
                    imgs = mmcv.imread(img_path)
                    res_bank.append({'boxes': boxes, 'imgs': imgs, 'out_path': out_path})
        return res_bank

    def tensor2idx(self, outputs):
        """
        Convert output tensor to text-index
        Args:
            outputs (tensor): model outputs with size: N * T * C
            img_metas (list[dict]): Each dict contains one image info.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]]
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                                         [0.9,0.9,0.98,0.97,0.96]]
        """
        batch_size = outputs.size(0)
        ignore_indexes = [37]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = outputs[idx, :, :]
            max_value, max_idx = torch.max(seq, -1)
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == 36:
                    break
                str_index.append(char_index)
                str_score.append(char_score)

            indexes.append(str_index)
            scores.append(str_score)

        return indexes, scores

    def reorderbox(self, bboxes, scores, texts):
        if len(bboxes) == 0:
            return bboxes, scores, texts
        boxes = bboxes[:, 0:8]
        ys = np.min(boxes[:, 1::2], axis=-1)
        ind = np.argsort(ys)
        bboxes = bboxes[ind]
        scores = scores[ind]
        texts = texts[ind]
        ys = ys[ind]

        thres = 10
        group = []
        for i, y in enumerate(ys):
            if i == 0:
                new = [y]
            elif i == (len(ys) - 1):
                if y - lasty < thres:
                    new.append(y)
                else:
                    group.append(new)
                    new = [y]
                group.append(new)
            else:
                if y - lasty < thres:
                    new.append(y)
                else:
                    group.append(new)
                    new = [y]
            lasty = y
        startid = 0
        for group1 in group:
            endid = startid + len(group1)
            groupbox = bboxes[startid:endid]
            groupscore = scores[startid:endid]
            grouptexts = texts[startid:endid]

            xs = np.min(groupbox[:, 0:8][:, 0::2], axis=-1)
            ind = np.argsort(xs)
            bboxes[startid:endid] = groupbox[ind]
            scores[startid:endid] = groupscore[ind]
            texts[startid:endid] = grouptexts[ind]

            startid = endid

        return bboxes, scores, texts

    def extractor(self, res_bank):
        for res in res_bank:
            bboxes = res['boxes']
            arr = res['imgs']
            out = res['out_path']

            char_scores = []
            texts = []
            filter_boxes = []
            for bbox in bboxes:
                assert len(bbox) == 9
                box = bbox[:8]
                box_score = bbox[8]
                box_img = crop_img(arr, box.tolist())
                w, h, _ = box_img.shape
                scale = w * h
                if not scale > 200:
                    continue
                recog_result = model_inference(self.recog_model, box_img)
                text = recog_result['text']
                char_score = recog_result['char_score']
                # process
                char_scores.append(char_score)
                texts.append(text)
                filter_boxes.append(bbox)
            char_scores = np.array(char_scores)
            filter_boxes = np.array(filter_boxes)
            texts = np.array(texts)
            filter_boxes, char_scores, texts = self.reorderbox(filter_boxes, char_scores, texts)
            final_feat = {'texts': texts, 'char_scores': char_scores,
                          'bboxes': filter_boxes}
            self.mkdir_savearray(path=out, array=final_feat)

    def ocr_feat(self,
                 img,
                 det_res,
                 output=None,
                 details=False,
                 export=None,
                 export_format='json',
                 batch_mode=False,
                 recog_batch_size=0,
                 det_batch_size=0,
                 single_batch_size=0,
                 imshow=False,
                 print_result=False,
                 merge=False,
                 merge_xdist=20,
                 **kwargs):

        res_bank = self.read_dets_imgs(det_res, img, output)
        print('totally {} img and det res read'.format(len(res_bank)))
        self.extractor(res_bank)



# Create an inference pipeline with parsed arguments
def main():
    args = parse_args()
    ocr = MMOCR(**vars(args))
    ocr.ocr_feat(**vars(args))


if __name__ == '__main__':
    main()
