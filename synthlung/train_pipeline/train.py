from monai.transforms import (Compose, LoadImaged)
import monai.config
import numpy as np

class TrainPipeline(object):
    monai.config.BACKEND = "Nibabel"
    def __init__(self) -> None:
        self.compose = Compose([
            LoadImaged(keys=['image', 'label'])
        ])
    
    def __call__(self, image_dict) -> None:
        for i in image_dict:
            self.compose(i)

    def verify_config(self) -> None:
        if (monai.config.deviceconfig.get_gpu_info()["Has CUDA"]):
            print(f"Running on: {monai.config.deviceconfig.get_gpu_info()['GPU 0 Name']}")
            return
        print("No CUDA found. Check config!")
        exit(1)