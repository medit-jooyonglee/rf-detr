import json
import tqdm
import cv2

import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pickle
from pathlib import Path

import torch.utils.data
import torchvision
import pycocotools.mask as coco_mask

import rfdetr.datasets.transforms as T
from trainer import diskmanager, jsonserialize, torch_utils
# from trainer.test.test_image_utils import bboxes
from trainer import utils_numpy

from rfdetr.datasets.coco import CocoDetection
# from .coco import (
#     CocoDetection, make_coco_transforms, make_coco_transforms_square_div_64
# )

def build(image_set, args, resolution):
    img_folder = str(args.dataset_dir)
    # annotation file is builed by functoin . see teeth.py::test_build_teethdsata
    annot_file = os.path.join(img_folder, 'teeth_coco.json')
    args_dict = dict(args.__dict__)

    dataset = TeethDetection(
        img_folder=img_folder,
        annot_file=annot_file,
        transforms=None,
        name=image_set,
        **args_dict
    )
    return dataset


class TeethDetection(CocoDetection):
    def __init__(self,
                 img_folder,
                 annot_file='',
                 transforms=None,
                 include_masks=False,
                 name='train',
                 debug=False,
                 split=None,
                 **kwargs):
        if os.path.exists(annot_file):
            super(TeethDetection, self).__init__(img_folder, annot_file, None, False)
        
        self._transforms = transforms
        self.include_masks = include_masks
        self.name = name
        self.debug = debug
        found = diskmanager.deep_search_all_files(
            str(img_folder), exts=['.pkl'])
        splits = {
            'train': [0, 0.85],
            'val': [0.85, 0.9],
            'valid': [0.85, 0.9],
            'test': [0.9, 1.0],
        }
        split = splits.get(name, split or [0, 1.0])

        found_files = []
        for path, files in found.items():
            if path.endswith('labels'):
                pass
            else:
                found_files.extend(files)

            # if path.endswith(name):
                # img_folder = path
        start, end = len(found_files) * split[0], len(found_files) * split[1]
        start, end = int(start), int(end)
        self.files = found_files[start:end]
        # self.prepare =
        # self

    def parse_file(self, idx, norm_bbox=True):
        file = self.files[idx]
        with open(file, 'rb') as f:

            data = pickle.load(f)

        input_img = data
        fname = os.path.basename(file).replace('.pkl', '.pkl')
        label_file = os.path.join(os.path.dirname(file), 'labels', fname)
        segment_file = os.path.join(os.path.dirname(
            file), 'segment', fname.replace('.pkl', '.png'))
        if not os.path.exists(label_file):
            raise FileNotFoundError(f'Annotation file not found: {label_file}')

        # if not os.path.
        segment_img = cv2.imread(segment_file, cv2.IMREAD_UNCHANGED)

        with open(label_file, 'rb') as f:
            annot = pickle.load(f)

        keys = ['teeth_bboxes', 'teeth_types', 'feature', 'fname']

        teeth_bboxes, teeth_types, features, fname = [
            annot[key] for key in keys]

        keys = list(teeth_bboxes.keys())

        bboxes, type_labels, fdi_labels = [], [], []
        for key in keys:
            bbox = teeth_bboxes[key]
            label = teeth_types[key]
            bboxes.append(bbox)
            type_labels.append(label)
            fdi_labels.append(int(key))
        bboxes = np.array(bboxes)
        type_labels = np.array(type_labels)
        bboxes = bboxes[..., :2].reshape([-1, 4])
        fdi_labels = np.array(fdi_labels)

        w, h = input_img.shape[:2]
        box_shape = np.array([w, h, w, h])
        if norm_bbox:
            bboxes = bboxes / box_shape

        vmin, vmax = data.min(), data.max()
        x0, y0, x1, y1 = np.split(bboxes, 4, axis=1)
        xc = (x0 + x1) / 2
        yc = (y0 + y1) / 2
        box_w = x1 - x0
        box_h = y1 - y0
        bboxes = np.concatenate([xc, yc, box_w, box_h], axis=1)
        # bboxes = utils_numpy.box_cxcywh_to_xyxy(bboxes)

        if self.include_masks:
            masks = fdi_labels.reshape(
                [fdi_labels.size, 1, 1]) == segment_img[None]
        
        
        
        if self.include_masks:
            
            area = masks.reshape(masks.shape[0], -1).sum(axis=1) if masks.size > 0 else np.zeros([0])
        else:
            box_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            area = box_area
            
        if bboxes.size == 0:
            print(f'empty{idx}')

        if self.debug:
            data_img = np.clip((input_img - vmin) /
                               (vmax - vmin) * 255.0, 0, 255)
            draw_bboxes(data_img, bboxes)
            cv2.imwrite('temp2.png', (data_img).astype(np.uint8))

            mask_img = draw_mask(data_img, masks)
            cv2.imwrite('temp_mask2.png', (mask_img).astype(np.uint8))

        shape = np.array(input_img.shape[:2])
        annot = dict(
            boxes=bboxes,
            labels=type_labels,
            image_id=np.array([idx]),
            area=area,
            is_crowd=np.zeros([bboxes.shape[0]], dtype=np.int64),
            # masks=masks,
            orig_size=shape,
            size=shape,
        )
        if self.include_masks:
            annot['masks'] = masks
            
        input_img = np.transpose(input_img, [2, 0, 1])

        return input_img, annot

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.parse_file(idx)
        img, target = torch_utils.data_convert(item, device='cpu')
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def coco_json_export(self, base_dir='', debug_break=-1):

        class_names = [
            'gum'
            'tooth',  # 1
            'inonlay',  # 2
            'crown',  # 3
            'bridge',  # 4
            'abutment',  # 5
            'laminate',  # 6
            'root-canal-treat',  # 7
            'residual-root',  # 8
        ]
        type_categories = [

        ]
        for i, name in enumerate(class_names):
            type_categories.append({
                "id": i,
                "name": name,
                'supercategory': 'tooth_type',
            })
        

        coco_json = {
            "images": [],
            "annotations": [],
            "categories": type_categories
        }

        annotations = coco_json["annotations"]
        images = coco_json["images"]
        annotation_id = 0
        for i in tqdm.tqdm(range(len(self))):
            item = self.parse_file(i, norm_bbox=False)
            filename = self.files[i]
            relative_filename = os.path.relpath(filename, base_dir)
            # annotation_id = 1
            image, target = item
            # image is CHW; shape[1:] == (H, W)
            height, width = image.shape[1:]
            image_id = i
            # for image_id, img in enumerate(custom_dataset, 1):?
            images.append({
                "id": image_id,
                "file_name": relative_filename,
                "width": width,
                "height": height,

            })
            keys = ['boxes', 'labels', 'area', 'is_crowd']
            bboxes, labels, areas, iscrowd = [target[key] for key in keys]

            for box, label, area, crowd in zip(bboxes, labels, areas, iscrowd):
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box.tolist(),  # [x, y, width, height]
                    "area": float(area),
                    "iscrowd": int(crowd),
                })
                annotation_id += 1
            if debug_break > 0 and debug_break > 20:
                # logg
                break

        return coco_json


def draw_bboxes(image, bboxes, colors=None, thickness=1):
    bboxes = np.asarray(bboxes)
    bboxes = bboxes.reshape([-1, 4])
    for i, box in enumerate(bboxes):
        if colors is not None:
            if len(colors) > i:
                color = colors[i]
            elif isinstance(colors, (list, tuple)) and len(colors) == 3:
                color = colors
        else:
            color = (0, 255, 0) # if colors is None else colors[i]
        color = tuple(map(int, color))
        p0, p1 =box[:2], box[2:]
        # p0 = np.round(p0).astype(np.int32)
        p0 = tuple(map(int, np.round(p0)))
        p1 = tuple(map(int, np.round(p1)))
        
        cv2.rectangle(image, p0, p1, color, thickness)


def test_coco():
    import pathlib

    img_folder = pathlib.Path('E:/dataset/coco/base/train2017')
    ann_file = pathlib.Path(
        'E:/dataset/coco/base/annotations/instances_train2017.json')
    from rfdetr.datasets.coco import CocoDetection
    data = CocoDetection(img_folder, ann_file, None, True)
    # data

    # from train
    img, target = data[0]

    target = torch_utils.to_numpy(target)
    img = np.array(img)

    bboxes = target['boxes']
    masks = target['masks']
    x0, y0, x1, y1 = np.split(bboxes, 4, axis=1)
    box_area = (np.abs(x1 - x0) * np.abs(y1 - y0)).reshape(-1)
    segment_area = [np.sum(m0) for m0 in masks]
    print(box_area, segment_area, target['area'])
    draw_bboxes(img, target['boxes'])
    plt.imsave('temp.png', img)
    target['boxes']

    draw_bboxes

    img2 = draw_mask(img, masks)
    plt.imsave('temp_mask.png', img2.astype(np.uint8))


def draw_mask(img, masks, alpha=0.5):
    for m0 in masks:
        img = utils_numpy.apply_mask(
            img, m0, color=np.random.uniform(0, 255, [3]), alpha=alpha)
    return img


def test_build_teethdsata():
    img_folder = 'E:/dataset/teeth_seg_3d/render_2dset2'
    ann_file = ''
    dataset = TeethDetection(img_folder, ann_file, None, True)
    assert len(dataset) > 0
    
    res = dataset.coco_json_export(base_dir=img_folder)
    
    # print(res)
    
    with open('teeth_coco.json', 'w') as f:
        json.dump(res, f)


def test_load():
    img_folder = 'E:/dataset/teeth_seg_3d/render_2dset2'
    ann_file = 'E:/dataset/teeth_seg_3d/render_2dset2/teeth_coco.json'
    dataset = TeethDetection(img_folder, ann_file, None, True)
    assert len(dataset) > 0
    
    # res = dataset.coco_json_export(base_dir=img_folder)
    # item = len(res)
    item = dataset[0]
    print(item)
    
    
if __name__ == "__main__":
    # test_coco()
    # test_build_teethdsata()
    test_load()