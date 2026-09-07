import tqdm
import os
import sys
sys.modules['torch'] = None
from trainer import diskmanager, image_utils
import shutil
import numpy as np


def test_main_copy():
    path = 'E:/dataset/reverse_tomosynthesis/cbct_ios_dcm'
    # base_save_dir = 'E:/dataset/temp/cbct_ios_dcm'
    base_save_dir = 'E:/dataset/Medit_AI Dropbox/Medit_AI Dropbox/Medit_AI/xray_panoramic/cbct_ios_dcm'
    found = diskmanager.deep_search_all_files(path, exts=['.jpg'])
    filter_image_size = [
        (1000, 2000)
    ]
    
    
    filter_files = []
    num_total_images = [len(files) for dirname, files in found.items()]
    num_total_images = np.sum(num_total_images)
    for dirname, files in tqdm.tqdm(found.items()):
        for file in files:
            img = image_utils.cv2_imread(file)
            
            shape = img.shape[:2]
            if np.all([np.all(shape > np.array(size0)) for size0 in filter_image_size]):
                # print(f"File: {file}, Shape: {shape}")
                filter_files.append(file)
                
    
    
    print(f"Total files found: {num_total_images} -> filtered files: {len(filter_files)}")
    
    for file in tqdm.tqdm(filter_files):
        relative_path = os.path.relpath(file, path)
        save_path = os.path.join(base_save_dir, relative_path)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy2(file, save_path)
    
test_main_copy()