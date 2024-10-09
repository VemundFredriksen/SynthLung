from typing import Any
from monai.transforms import (Compose, LoadImaged, SaveImaged)
import monai.config
import numpy as np
import tqdm

class TumorSeedIsolationd(object):
    def __init__(self, image_key='image', label_key='label', image_output_key='seed_image', label_output_key='seed_label') -> None:
        self.image_key = image_key
        self.label_key = label_key
        self.image_output_key = image_output_key
        self.label_output_key = label_output_key

    def __call__(self, sample: dict) -> dict:
        image, label = sample[self.image_key], sample[self.label_key]
        bolean_mask = (label > 0).astype(np.uint8)

        indeces = np.argwhere(bolean_mask)
        y_min, x_min, z_min = indeces.min(axis=0)
        y_max, x_max, z_max = indeces.max(axis=0)

        clipped_image = image[y_min:y_max+1, x_min:x_max+1, z_min:z_max+1]
        clipped_label = label[y_min:y_max+1, x_min:x_max+1, z_min:z_max+1]
        clipped_image = np.where(clipped_label == 1, clipped_image, -1024)

        sample[self.image_output_key] = clipped_image
        sample[f'{self.image_output_key}_meta_dict'] = sample[f'{self.image_key}_meta_dict']
        self.__update_image_dims__(sample[f'{self.image_output_key}_meta_dict'], clipped_image.shape)

        sample[self.label_output_key] = clipped_label
        sample[f'{self.label_output_key}_meta_dict'] = sample[f'{self.label_key}_meta_dict']
        self.__update_image_dims__(sample[f'{self.label_output_key}_meta_dict'], clipped_label.shape)

        return sample
    
    def __update_image_dims__(self, meta_dict: dict, new_dims: tuple) -> None:
        meta_dict['size'] = new_dims
        meta_dict['dim'][1] = new_dims[0]
        meta_dict['dim'][2] = new_dims[1]
        meta_dict['dim'][3] = new_dims[2]
        meta_dict['spatial_shape'][0] = new_dims[0]
        meta_dict['spatial_shape'][1] = new_dims[1]
        meta_dict['spatial_shape'][2] = new_dims[2]

    def __update_image_filename__(self, image):
        image['filename_or_obj'] = image['filename_or_obj'].replace('source_', 'seed_')
    
    def __update_label_filename__(self, label):
        label.meta['filename_or_obj'] = label.meta['filename_or_obj'].replace('source_', 'seed_')

class RenameSourceToSeed(object):
    def __init__(self, meta_dict_keys=['seed_image_meta_dict'], image_object_keys=['seed_label']) -> Any:
        self.meta_dict_keys= meta_dict_keys
        self.image_object_keys = image_object_keys

    def __call__(self, sample:dict) -> Any:
        for meta_dict in self.meta_dict_keys:
            sample[meta_dict]['filename_or_obj'] = sample[meta_dict]['filename_or_obj'].replace('source_', 'seed_')
        
        for image_object in self.image_object_keys:
            sample[image_object].meta['filename_or_obj'] = sample[image_object].meta['filename_or_obj'].replace('source_', 'seed_')

        return sample
    

class TumorCropPipeline(object):
    def __init__(self) -> None:
        self.compose = Compose([
            LoadImaged(keys=['image', 'label'], image_only = False),
            TumorSeedIsolationd(image_key='image', label_key='label', image_output_key='seed_image', label_output_key='seed_label'),
            RenameSourceToSeed(meta_dict_keys=['seed_image_meta_dict', 'seed_label_meta_dict']),
            SaveImaged(keys=['seed_image'], output_dir='./assets/seeds/', output_postfix='', separate_folder=False, writer=monai.data.NibabelWriter),
            SaveImaged(keys=['seed_label'], output_dir='./assets/seeds/', output_postfix='', separate_folder=False, writer=monai.data.NibabelWriter)
        ])
    
    def __call__(self, image_dict) -> None:
        if isinstance(image_dict, list):
            print(f"Tumor isolation for {len(image_dict)} images starting...")
            for sample in tqdm.tqdm(image_dict):
                self.compose(sample)
        else:
            self.compose(image_dict)
        
        print("Tumor isolation complete!")


