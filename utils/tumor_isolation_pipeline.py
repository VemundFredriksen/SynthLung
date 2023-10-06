from monai.transforms import (Compose, LoadImaged, SaveImaged)
import numpy as np

image_dict = [
        {
            'image' : ".\\assets\\Task06_Lung\\imagesTr\\lung_001.nii.gz",
            'label' : ".\\assets\\Task06_Lung\\labelsTr\\lung_001.nii.gz"
        },
        {
            'image' : ".\\assets\\Task06_Lung\\imagesTr\\lung_003.nii.gz",
            'label' : ".\\assets\\Task06_Lung\\labelsTr\\lung_003.nii.gz"
        }
    ]

class ClipAroundMask(object):
    def __call__(self, sample):
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
    
    def __update_image_dims__(self, meta_dict, new_dims):
        meta_dict['size'] = new_dims
        meta_dict['dim'][1] = new_dims[0]
        meta_dict['dim'][2] = new_dims[1]
        meta_dict['dim'][3] = new_dims[2]
        meta_dict['spatial_shape'][0] = new_dims[0]
        meta_dict['spatial_shape'][1] = new_dims[1]
        meta_dict['spatial_shape'][2] = new_dims[2]

clipAroundPipeline= Compose(
    [
        LoadImaged(keys=['image', 'label']),
        ClipAroundMask(),
        SaveImaged(keys=['image'], output_dir='./out', output_postfix='_image.nii.gz'),
        SaveImaged(keys=['label'], output_dir='./out', output_postfix='_label.nii.gz')
    ]
)

clipAroundPipeline(image_dict)

