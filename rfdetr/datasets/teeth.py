import json
import cv2

import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pickle
from pathlib import Path


import pycocotools.mask as coco_mask

import rfdetr.datasets.transforms as T
from trainer import diskmanager, jsonserialize, torch_utils
# from trainer.test.test_image_utils import bboxes
from trainer import utils_numpy



def build(image_set, args, resolution):
    img_folder = str(args.dataset_dir)

    args_dict = dict(args.__dict__)
    
    dataset = TeethDetection(
        img_folder=img_folder, 
        transforms=None,
        name=image_set,
        **args_dict
    )
    return dataset



class TeethDetection:
    def __init__(self, *,
                 img_folder, 
                 transforms=None, include_masks=False, name='train', debug=False,**kwargs):
        super(TeethDetection, self).__init__()
        self._transforms = transforms
        self.include_masks = include_masks
        self.name = name
        self.debug = debug
        found = diskmanager.deep_search_all_files(str(img_folder), exts=['.pkl'])
        
        found_files = []
        for path, files in found.items():
            if path.endswith('labels'):
                pass
            else:
                found_files.extend(files)
                
            # if path.endswith(name):
                # img_folder = path
        self.files = found_files
        # self

    def parse_file(self, idx):
        file = self.files[idx]
        with open(file, 'rb') as f:
        
            data = pickle.load(f)
        
        input_img = data
        fname = os.path.basename(file).replace('.pkl', '.pkl')
        label_file = os.path.join(os.path.dirname(file), 'labels', fname)
        segment_file = os.path.join(os.path.dirname(file), 'segment', fname.replace('.pkl', '.png'))
        if not os.path.exists(label_file):
            raise FileNotFoundError(f'Annotation file not found: {label_file}')
        
        # if not os.path.
        segment_img = cv2.imread(segment_file, cv2.IMREAD_UNCHANGED)
        
        
        with open(label_file, 'rb') as f:
            annot = pickle.load(f)
            
    
        keys = ['teeth_bboxes', 'teeth_types', 'feature', 'fname']
        
        teeth_bboxes, teeth_types, features, fname = [annot[key] for key in keys]
        
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
        
        
        vmin, vmax = data.min(), data.max()
        
        masks = fdi_labels.reshape([fdi_labels.size, 1, 1]) == segment_img[None]
        area = masks.reshape(masks.shape[0], -1).sum(axis=1)
        
        if self.debug:
            data_img = np.clip((input_img - vmin) / (vmax - vmin) * 255.0, 0, 255)
            draw_bboxes(data_img, bboxes)
            cv2.imwrite('temp2.png', (data_img).astype(np.uint8))
            
            mask_img = draw_mask(data_img, masks)
            cv2.imwrite('temp_mask2.png', (mask_img).astype(np.uint8))
            
        shape = np.array(input_img.shape[:2])
        annot = dict(
            boxes = bboxes,
            labels = type_labels,
            image_id = np.array([idx]),
            area=area,
            is_crowd=np.zeros([bboxes.shape[0]], dtype=np.int64), 
            masks=masks,
            orig_size=shape,
            size=shape,
        )
        
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
            


def draw_bboxes(image, bboxes):
    bboxes = np.asarray(bboxes)
    bboxes = bboxes.reshape([-1, 4])
    for box in bboxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 2)
        
def test_coco():
    import pathlib 
    
    img_folder = pathlib.Path('E:/dataset/coco/base/train2017')
    ann_file = pathlib.Path('E:/dataset/coco/base/annotations/instances_train2017.json')
    from rfdetr.datasets.coco import CocoDetection
    data = CocoDetection(img_folder, ann_file, None, True)
    # data

    # from train
    img, target = data[0]


    target = torch_utils.to_numpy(target)
    img = np.array(img)
    
    bboxes  = target['boxes']
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
        img = utils_numpy.apply_mask(img, m0, color=np.random.uniform(0, 255, [3]), alpha=alpha)
    return img

     
def test_build_teethdsata():
    img_folder = 'E:/dataset/teeth_seg_3d/render_2dset2'
    ann_file = ''
    datset = TeethDetection(img_folder, ann_file, None, True)
    assert len(datset) > 0
    
    for _ in range(len(datset)):
        try:
            img, target = datset[_]
            # print(img.size, target)
        except Exception as e:
            print(f'Error at {_}: {e}')
            continue
        # import numpy as np
        # import matplotlib.pyplot as plt

        # img = np.array(img)
        # plt.imsave(f'temp_{_}.png', img)
        break

# plt.imshow(np.array(img))

if __name__ == "__main__":
    # test_coco()
    test_build_teethdsata()