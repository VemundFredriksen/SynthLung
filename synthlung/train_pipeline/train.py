import monai.config
import json

from synthlung.dataset.synthlung_dataset import SynthlungDataset
from synthlung.providers.loss_function_provider import LossFunctionProvider
from synthlung.providers.network_provider import NetworkProvider
from synthlung.providers.optimizer_provider import OptimizerProvider
from synthlung.train_pipeline.trainer import Trainer
from torch.utils.data import DataLoader

class TrainPipeline():
    monai.config.BACKEND = "Nibabel"
    def __init__(self, provider) -> None:
        self.provider = provider
        self.config = self.provider.get_config()
        self.loss_function_provider = LossFunctionProvider(self.config)
        self.model_provider = NetworkProvider(self.config)
        self.optimizer_provider = OptimizerProvider(self.config)

        model = self.model_provider()
        model.to(device='cuda')

        self.trainer = Trainer(model, self.optimizer_provider(self.model_provider()), self.loss_function_provider(), self.config["modelSavePath"])

        with open(self.config["trainImagesDatasetPath"], "r") as f:
            self.data = json.load(f)
        self.dataset = SynthlungDataset(self.data)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)

    def __call__(self) -> None:
        self.trainer(self.dataloader, 1)

    def verify_config(self) -> None:
        if (monai.config.deviceconfig.get_gpu_info()["Has CUDA"]):
            print(f"Running on: {monai.config.deviceconfig.get_gpu_info()['GPU 0 Name']}")
            return
        print("No CUDA found. Check config!")
        exit(1)