from typing import Any
from monai.transforms import (Compose, LoadImaged, SaveImaged)
import numpy as np

class CutOutTumor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        bolean_mask = (label > 0).astype(np.uint8)

        indeces = np.argwhere(bolean_mask)
        y_min, x_min, z_min = indeces.min(axis=0)
        y_max, x_max, z_max = indeces.max(axis=0)

        clipped_image = image[y_min:y_max+1, x_min:x_max+1, z_min:z_max+1]
        clipped_label = label[y_min:y_max+1, x_min:x_max+1, z_min:z_max+1]

        offset_location = (np.random.rand(3) * (sample['image_meta_dict']['spatial_shape'] - clipped_image.shape)).astype(int)

        image[offset_location[0] : offset_location[0] + clipped_image.shape[0], offset_location[1] : offset_location[1] + clipped_image.shape[1], offset_location[2] : offset_location[2] + clipped_image.shape[2]][clipped_label == 1] = clipped_image[clipped_label == 1]
        label[offset_location[0] : offset_location[0] + clipped_image.shape[0], offset_location[1] : offset_location[1] + clipped_image.shape[1], offset_location[2] : offset_location[2] + clipped_image.shape[2]][clipped_label == 1] = clipped_label[clipped_label == 1]

        return sample
    
    def __update_image_dims__(self, meta_dict, new_dims):
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
            SaveImaged(keys=['image'], output_dir='./assets/isolated_tumors', output_postfix='_image'),
            SaveImaged(keys=['label'], output_dir='./assets/isolated_tumors', output_postfix='_label')
        ])
    
    def __call__(self, image_dict) -> None:
        self.compose(image_dict)

