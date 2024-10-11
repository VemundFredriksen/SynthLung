from typing import Any
from monai.transforms import (Compose, LoadImaged, SaveImaged, ToMetaTensord)
from lungmask import LMInferer
from synthlung.utils.json_generator import JSONGenerator
import tqdm
import os
import json

NII_GZ_EXTENSION = '.nii.gz'
IMAGE_NII_GZ = 'image.nii.gz'
LABEL_NII_GZ = 'label.nii.gz'

class MaskLungs(object):
    def __call__(self, sample) -> Any:
        image = sample['image']
        image = image.numpy()

        image = self.__transpose_for_lungmask__(image)
        
        mask = self.lungmask_inferer.apply(image)
        mask = self.__transpose_for_lungmask__(mask)

        sample['mask'] = mask

        sample['mask_meta_dict'] = sample['image_meta_dict']
        self.__adjust_filename__(sample['mask_meta_dict'])

        return sample

    def __init__(self, lungmask_inferer: LMInferer) -> None:
        self.lungmask_inferer = lungmask_inferer

    def __transpose_for_lungmask__(self, image):
        transpose_indeces = (2, 1, 0)
        return image.transpose(transpose_indeces)

    def __adjust_filename__(self, mask_metadict):
        mask_metadict['filename_or_obj'] = mask_metadict['filename_or_obj'].replace('source_', 'host_').replace('_image', '_label')

class LungMaskPipeline(object):
    def __init__(self, lungmask_inferer: LMInferer) -> None:
        self.inferer = lungmask_inferer
        self.compose = Compose([
            LoadImaged(keys=['image'], image_only = False),
            MaskLungs(lungmask_inferer=self.inferer),
            SaveImaged(keys=['mask'], output_dir='./assets/images/hosts/', output_postfix='', separate_folder=False)
        ])

    def __call__(self, image_dict) -> Any:
        if isinstance(image_dict, list):
            print(f"Lung masking for {len(image_dict)} images starting...")
            for sample in tqdm.tqdm(image_dict):
                self.compose(sample)
        else:
            self.compose(image_dict)

class HostJsonGenerator(JSONGenerator):
    def __init__(self, path) -> None:
        self.path = path

    def generate_json(self) -> None:
        dataset_json = []
        for filename in os.listdir(self.path):
            if filename.endswith((NII_GZ_EXTENSION)):
                sample_data = {
                    "host_image": "./assets/images/hosts/" + (filename[:filename.index(LABEL_NII_GZ)] + IMAGE_NII_GZ).replace('host_', 'source_'),
                    "host_label": self.path + filename
                }
                dataset_json.append(sample_data)

        with open(self.path + "/dataset.json", 'w') as json_file:
            json.dump(dataset_json, json_file, indent=4)