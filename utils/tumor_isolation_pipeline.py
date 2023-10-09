from typing import Any
from monai.transforms import (Compose, LoadImaged, SaveImaged)
import numpy as np
import tqdm

class CutOutTumor(object):
    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']
        bolean_mask = (label > 0).astype(np.uint8)

        indeces = np.argwhere(bolean_mask)
        y_min, x_min, z_min = indeces.min(axis=0)
        y_max, x_max, z_max = indeces.max(axis=0)

        clipped_image = image[y_min:y_max+1, x_min:x_max+1, z_min:z_max+1]
        clipped_label = label[y_min:y_max+1, x_min:x_max+1, z_min:z_max+1]
        clipped_image = np.where(clipped_label == 1, clipped_image, -1024)

        sample['image'] = clipped_image
        self.__update_image_dims__(sample['image_meta_dict'], clipped_image.shape)

        sample['label'] = clipped_label
        self.__update_image_dims__(sample['label_meta_dict'], clipped_label.shape)

        return sample
    
    def __update_image_dims__(self, meta_dict: dict, new_dims: tuple) -> None:
        meta_dict['size'] = new_dims
        meta_dict['dim'][1] = new_dims[0]
        meta_dict['dim'][2] = new_dims[1]
        meta_dict['dim'][3] = new_dims[2]
        meta_dict['spatial_shape'][0] = new_dims[0]
        meta_dict['spatial_shape'][1] = new_dims[1]
        meta_dict['spatial_shape'][2] = new_dims[2]

class TumorCropPipeline(object):
    def __init__(self) -> None:
        self.compose = Compose([
            LoadImaged(keys=['image', 'label']),
            CutOutTumor(),
            SaveImaged(keys=['image'], output_dir='./assets/seeds/', output_postfix='_image', separate_folder=False),
            SaveImaged(keys=['label'], output_dir='./assets/seeds/', output_postfix='_label', separate_folder=False)
        ])
    
    def __call__(self, image_dict) -> None:
        if isinstance(image_dict, list):
            print(f"Tumor isolation for {len(image_dict)} images starting...")
            for sample in tqdm.tqdm(image_dict):
                self.compose(sample)
        else:
            self.compose(image_dict)
        
        print(f"Tumor isolation complete!")


