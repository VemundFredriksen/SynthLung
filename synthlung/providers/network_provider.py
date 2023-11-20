from synthlung.networks.simple_network import SimpleNN
from monai.networks.nets.basic_unet import BasicUNet

class NetworkProvider():
    def __init__(self, config) -> None:
        if (config["network"] == "SimpleNN"):
            self.network = SimpleNN()
        elif(config['network'] == "BasicUnet"):
            self.network = BasicUNet(spatial_dims=3, in_channels=1, out_channels=1)

    def __call__(self):
        return self.network