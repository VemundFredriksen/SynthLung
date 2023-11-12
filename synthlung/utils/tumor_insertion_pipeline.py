from typing import Any
from monai.transforms import (Compose, LoadImaged, SaveImaged)
import numpy as np
import datetime
import random
import json
import os
import requests
import secrets

seed = 3

class InsertTumor(object):
    def __call__(self, sample) -> None:
        image, label, image_mask, seed_image, seed_label = sample['image'], sample['label'], sample["image_mask"], sample['seed_image'], sample['seed_label']

        offset_randomizer = np.random.default_rng(seed)

        while (True):
            offset_location = (offset_randomizer.random(size=3) * sample['image_meta_dict']['spatial_shape']).astype(int)
            if (image_mask[offset_location[0], offset_location[1], offset_location[2]] > 0):
                break

        offset_location = offset_location - np.array(seed_image.shape) // 2

        image[offset_location[0] : offset_location[0] + seed_image.shape[0], offset_location[1] : offset_location[1] + seed_image.shape[1], offset_location[2] : offset_location[2] + seed_image.shape[2]][seed_label == 1] = seed_image[seed_label == 1]
        label[offset_location[0] : offset_location[0] + seed_image.shape[0], offset_location[1] : offset_location[1] + seed_image.shape[1], offset_location[2] : offset_location[2] + seed_image.shape[2]][seed_label == 1] = 1

        sample["image_meta_dict"]["filename_or_obj"] = sample["randomized_image"]
        sample["label_meta_dict"]["filename_or_obj"] = sample["randomized_label"]

        return sample

class InsertTumorPipeline(object):
    def __init__(self) -> None:
        self.randomized_dict = []
        self.time = f"{datetime.datetime.now()}".replace(" ", "-").replace(":", ".")
        self.dir_name = f"./assets/images/artificial_tumors/{self.time}/"
        self.compose = Compose([
            LoadImaged(keys=['image', 'label', 'image_mask', 'seed_image', 'seed_label']),
            InsertTumor(),
            SaveImaged(keys=['image'], output_dir=f"{self.dir_name}randomzied_images/", output_postfix="", separate_folder=False),
            SaveImaged(keys=['label'], output_dir=f"{self.dir_name}randomzied_images/", output_postfix="", separate_folder=False)
        ])
    
    def getDict(self):
        return self.randomized_dict

    def __call__(self, image_dict, seeds_dict) -> None:
        self._generate_randomized_dict(image_dict, seeds_dict)
        self.compose(self.randomized_dict)

    def _generate_randomized_dict(self, image_dict, seeds_dict):
        for n, i in enumerate(range(2)):
            image = random.choice(image_dict)
            seed = random.choice(seeds_dict)

            image["image_mask"] = image["label"].replace("/source/", "/hosts/").replace("source_", "host_")

            output_name = {"randomized_image" : f"{self.dir_name}randomzied_images/{n}_image.nii.gz",
                           "randomized_label" : f"{self.dir_name}randomzied_images/{n}_label.nii.gz"}

            self.randomized_dict.append({**image, **seed, **output_name})

        self._log()

    def _log(self):
        log_name = self.dir_name + "log/"
            
        if not os.path.exists(log_name):
            os.makedirs(log_name)

        with open(log_name + "dataset.json", 'w') as json_file:
            json.dump(self.randomized_dict, json_file, indent=4)

        if (True):
            return

        for elem in self.randomized_dict:
            jsonData = {"id" : str(secrets.token_hex(12))}
            jsonData = {**jsonData, **elem}
            jsonData = json.dumps(jsonData)
            url = "http://localhost:5167/api/Log"
            header = {"Content-Type" : "application/json"}

            print(jsonData)
    
            requests.post(url, data=jsonData, headers=header, verify=False)
