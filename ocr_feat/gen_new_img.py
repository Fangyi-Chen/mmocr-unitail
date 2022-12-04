'''
generate new images by blurring/zeroing the non-text region
input:  unitail-cls-aug
        det_res_aug
output: unitail-cls-blur / unitail-cls-zero
Fangyi Chen @ CMU
Feb 14 2022
'''
import os
import os.path as osp
import numpy as np
import mmcv
import cv2


def mkdir_savearray(path, array):
    folder = osp.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(path, array)

def read_dets_imgs(det_res, img, output):
    res_bank = []
    train_or_test = ['train', 'test', 'val']
    for t in train_or_test:
        classes = os.listdir(osp.join(det_res, t))
        for cls in classes:
            for txt in os.listdir(osp.join(det_res, t, cls)):
                txt_path = osp.join(det_res, t, cls, txt)
                img_path = osp.join(img, t, cls, txt[:-4] + '.jpg')
                out_path = osp.join(output, t, cls, txt[:-4] + '.jpg')
                f = open(txt_path, "r")
                boxes = np.array([[float(b) for b in box.strip().split(',')] for box in f.readlines()])
                f.close()
                imgs = mmcv.imread(img_path)
                res_bank.append({'boxes': boxes, 'imgs': imgs, 'out_path': out_path})
    return res_bank


def encoder_circumscribed(bboxes):
    xmin = np.min(bboxes[:, 0::2], axis=1).reshape(-1, 1)
    ymin = np.min(bboxes[:, 1::2], axis=1).reshape(-1, 1)
    xmax = np.max(bboxes[:, 0::2], axis=1).reshape(-1, 1)
    ymax = np.max(bboxes[:, 1::2], axis=1).reshape(-1, 1)
    return np.concatenate((xmin, ymin, xmax, ymax), axis=1)


def proc(res_bank, type='zero'):
    new_bank = []
    for res in res_bank:
        img = res['imgs']
        if len(res['boxes']) == 0:
            boxes = np.zeros((0, 8))
        else:
            boxes = res['boxes'][:, :8]
        circums = encoder_circumscribed(boxes)
        h, w, _ = img.shape
        if type == 'zero':
            patch = np.zeros_like(img)
        elif type == 'blur':
            patch = cv2.GaussianBlur(img, (9, 9), cv2.BORDER_DEFAULT)

        for circum in circums:
            xmin, ymin, xmax, ymax = circum
            xmin, ymin = max(int(xmin), 0), max(int(ymin), 0)
            xmax, ymax = min(int(xmax), w-1), min(int(ymax), h-1)
            patch[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]
        save_path = res['out_path']
        mkdir_savearray(save_path, patch)
        # cv2.imshow('1', patch)
        # cv2.waitKey(0)
        new_bank.append(patch)
    return new_bank



# Create an inference pipeline with parsed arguments
def main():
    img_root = 'ocr_feat/unitail-cls-aug'
    det_root = 'ocr_feat/det_res_aug/out_txt_dir'
    output   = 'ocr_feat/unitail-cls-aug-proc'
    res_bank = read_dets_imgs(det_root, img_root, output)
    new_bank = proc(res_bank, type='blur')


if __name__ == '__main__':
    main()


