from torch.utils.data import Dataset
from monai.transforms import (Compose, LoadImaged, ToTensord)
from synthlung.utils.send_to_cudad import SendToCudad

class SynthlungDataset(Dataset):
    def __init__(self, data: [dict]):
        self.data = data
        self.compose_load = Compose([
            LoadImaged(keys=['image', 'label']),
            ToTensord(keys=['image', 'label']),
            SendToCudad(keys=['image', 'label'])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        loaded_data = self.compose_load(self.data[index])
        return loaded_data["image"], loaded_data["label"]