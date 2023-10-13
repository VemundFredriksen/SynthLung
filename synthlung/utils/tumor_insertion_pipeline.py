from typing import Any
from monai.transforms import (Compose, LoadImaged, SaveImaged)
import numpy as np
import datetime
import random
import json
import os

class InsertTumor(object):
    def __call__(self, sample) -> None:
        image, label, seed_image, seed_label = sample['image'], sample['label'], sample['seed_image'], sample['seed_label']

        lungmask = self.__generate_random_lungmask__(image)

        while (True): #Temporary offset calculation
            offset_location = (np.random.rand(3) * (sample['image_meta_dict']['spatial_shape'] - seed_image.shape)).astype(int)
            if (lungmask[offset_location[0], offset_location[1], offset_location[2]] == 1):
                break

        image[offset_location[0] : offset_location[0] + seed_image.shape[0], offset_location[1] : offset_location[1] + seed_image.shape[1], offset_location[2] : offset_location[2] + seed_image.shape[2]][seed_label == 1] = seed_image[seed_label == 1]
        label[offset_location[0] : offset_location[0] + seed_image.shape[0], offset_location[1] : offset_location[1] + seed_image.shape[1], offset_location[2] : offset_location[2] + seed_image.shape[2]][seed_label == 1] = 1

        self.__remove_filename__(image)
        self.__remove_filename__(label)

        return sample
    
    def __generate_random_lungmask__(self, image): #Temporary method
        shape_dim_0 = image.shape[0]
        shape_dim_1 = image.shape[1]
        shape_dim_2 = image.shape[2]
        dim0min = shape_dim_0 // 4
        dim0max = 3 * dim0min
        dim1min = shape_dim_1 // 4
        dim1max = 3 * dim1min
        dim2min = shape_dim_2 // 4
        dim2max = 3 * dim2min
        arra = np.where(image > 0, 0, 0)
        arra[dim0min : dim0max, dim1min : dim1max, dim2min : dim2max] = 1
        return arra
    
    def __remove_filename__(self, image):
        image.meta['filename_or_obj'] = ""

class InsertTumorPipeline(object):
    def __init__(self) -> None:
        self.randomized_dict = []
        self.time = f"{datetime.datetime.now()}".replace(" ", "-").replace(":", ".")
        self.dir_name = f"./assets/artificial_tumors/{self.time}/"
        self.compose = Compose([
            LoadImaged(keys=['image', 'label', 'seed_image', 'seed_label']),
            InsertTumor(),
            SaveImaged(keys=['image'], output_dir=f"{self.dir_name}randomzied_images/", output_postfix=f"image"),
            SaveImaged(keys=['label'], output_dir=f"{self.dir_name}randomzied_images/", output_postfix=f"label")
        ])
    
    def __call__(self, image_dict, seeds_dict) -> None:
        self.__generate_randomized_dict__(image_dict, seeds_dict)
        self.compose(self.randomized_dict)

    def __generate_randomized_dict__(self, image_dict, seeds_dict):
        for n, i in enumerate(range(1)):
            image = random.choice(image_dict)
            seed = random.choice(seeds_dict)

            output_name = {"randomized_image" : f"{self.dir_name}randomzied_images/image_{n}.nii.gz",
                           "randomized_label" : f"{self.dir_name}randomzied_images/label_{n}.nii.gz"}

            self.randomized_dict.append({**image, **seed, **output_name})
        
        self.__log__()

    def __log__(self):
        log_name = self.dir_name + "log/"
            
        if not os.path.exists(log_name):
            os.makedirs(log_name)

        with open(log_name + "dataset.json", 'w') as json_file:
            json.dump(self.randomized_dict, json_file, indent=4)
