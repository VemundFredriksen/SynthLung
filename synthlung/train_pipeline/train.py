import monai.config
import json

from synthlung.dataset.synthlung_dataset import CustomDataset
from synthlung.providers.loss_function_provider import LossFunctionProvider
from synthlung.providers.network_provider import NetworkProvider
from synthlung.providers.optimizer_provider import OptimizerProvider
from synthlung.train_pipeline.trainer import Trainer
from torch.utils.data import DataLoader

class TrainPipeline():
    monai.config.BACKEND = "Nibabel"
    def __init__(self, config) -> None:
        self.config = config
        self.loss_function_provider = LossFunctionProvider(config)
        self.model_provider = NetworkProvider(config)
        self.optimizer_provider = OptimizerProvider(config)

        model = self.model_provider()
        model.to(device='cuda')

        self.trainer = Trainer(model, self.optimizer_provider(self.model_provider()), self.loss_function_provider(), self.config["model_save_path"])

        with open(config["train_images_dataset_path"], "r") as f:
            self.data = json.load(f)
        self.dataset = CustomDataset(self.data)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)

    def __call__(self) -> None:
        self.trainer(self.dataloader, 1)

    def verify_config(self) -> None:
        if (monai.config.deviceconfig.get_gpu_info()["Has CUDA"]):
            print(f"Running on: {monai.config.deviceconfig.get_gpu_info()['GPU 0 Name']}")
            return
        print("No CUDA found. Check config!")
        exit(1)